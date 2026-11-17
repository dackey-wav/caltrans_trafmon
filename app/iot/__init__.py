"""IoT demo: a sensor node replays station data over MQTT, a predictor node forecasts it.

Two logical devices talk only through an MQTT broker:

    sensor node  --traffic/pems/stations/<id>/5min-------->  predictor node
    sensor node  <--traffic/pems/stations/<id>/control-----  (operator / dashboard)
    predictor    --traffic/pems/stations/<id>/forecast---->  dashboard
    predictor    --traffic/pems/stations/<id>/evaluation-->  dashboard
    both         --.../status (retained, Last Will)------->  everyone

The 5min message schema is the one from ``app.live.feed.Measurement`` so the
live PeMS FTP publisher and the replay emulator are interchangeable sources.
"""
