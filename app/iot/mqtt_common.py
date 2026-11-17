"""MQTT client setup shared by the IoT nodes (paho-mqtt 2.x, MQTT v3.1.1)."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable

import paho.mqtt.client as mqtt

LOG = logging.getLogger("mqtt")


def connect_client(
    client_id: str,
    *,
    will: tuple[str, str] | None = None,
    birth: tuple[str, str | Callable[[], str]] | None = None,
    subscriptions: list[tuple[str, int]] | None = None,
    clean_session: bool = True,
    manual_ack: bool = False,
    on_message=None,
    timeout: float = 60.0,
) -> mqtt.Client:
    """Connect, start the network loop and wait for CONNACK.

    ``subscriptions`` are (re)sent on every connect so a broker restart does
    not silently stop the data flow. ``will`` is published retained by the
    broker if this node disappears without a clean disconnect; ``birth`` is
    the matching retained "online" message, sent on every connect.
    """
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=client_id,
        clean_session=clean_session,
        manual_ack=manual_ack,
    )
    if on_message is not None:
        client.on_message = on_message  # set before connect: queued messages arrive at once
    user = os.getenv("MQTT_USERNAME")
    if user:
        client.username_pw_set(user, os.getenv("MQTT_PASSWORD") or None)
    if os.getenv("MQTT_TLS") == "1":
        client.tls_set(ca_certs=os.getenv("MQTT_CA_CERT") or None)
    if will:
        client.will_set(will[0], will[1], qos=1, retain=True)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    connected = threading.Event()
    subs = list(subscriptions or [])

    def on_connect(c, _userdata, _flags, reason_code, _properties):
        if reason_code.is_failure:
            LOG.error("broker refused connection: %s", reason_code)
            return
        for topic, qos in subs:
            c.subscribe(topic, qos=qos)
        if birth:
            # Re-announce on every connect: a session takeover after a crash
            # makes the broker publish the old connection's Will ("offline")
            # after we are already back.
            payload = birth[1]() if callable(birth[1]) else birth[1]
            c.publish(birth[0], payload, qos=1, retain=True)
        LOG.info("connected to broker as %s; subscribed %s", client_id, [t for t, _ in subs])
        connected.set()

    def on_disconnect(_c, _userdata, _flags, reason_code, _properties):
        LOG.warning("disconnected from broker: %s (auto-reconnect)", reason_code)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    host = os.getenv("MQTT_HOST", "localhost")
    port = int(os.getenv("MQTT_PORT", "1883"))
    client.connect_async(host, port, keepalive=30)
    client.loop_start()
    if not connected.wait(timeout):
        client.loop_stop()
        raise ConnectionError(f"no MQTT broker at {host}:{port} after {timeout:.0f}s")
    return client


def go_offline(client: mqtt.Client) -> None:
    """Simulated network outage: leave the broker and stop the network loop."""
    client.disconnect()
    client.loop_stop()


def go_online(client: mqtt.Client) -> None:
    """End a simulated outage; on_connect re-subscribes and re-announces the node."""
    while True:
        try:
            client.reconnect()
            break
        except OSError as exc:
            LOG.warning("reconnect failed (%s); retrying", exc)
            time.sleep(2)
    client.loop_start()
