"""Dispatch handlers for TimesNet commands."""

from models.TimesNet import config as tc


def _common_train_kwargs(args) -> dict:
    return dict(
        patience=args.patience,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        use_amp=not args.no_amp,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        oom_retry_enabled=not args.no_oom_retry,
        oom_retry_min_batch=args.oom_retry_min_batch,
        lr_scheduler_enabled=not args.no_lr_scheduler,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
    )


def dispatch(args, parser) -> int | None:
    if args.command != "timesnet":
        return None

    if args.timesnet_command == "prepare":
        from models.TimesNet.runner import run_prepare

        run_prepare(
            run_id=args.run_id or tc.default_run_id(),
            date_from=args.date_from,
            date_to=args.date_to,
            freq=args.freq,
        )
        return 0

    if args.timesnet_command == "train":
        from models.TimesNet.runner import run_train

        run_train(
            args.run_id,
            name=args.name,
            window=args.window,
            pred_len=args.pred_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            d_model=args.d_model,
            d_ff=args.d_ff,
            e_layers=args.e_layers,
            top_k=args.top_k,
            num_kernels=args.num_kernels,
            dropout=args.dropout,
            seed=args.seed,
            loss=args.loss,
            mark_set=args.mark_set,
            target_mode=args.target,
            train_stride=args.train_stride,
            evaluate=not args.no_eval,
            **_common_train_kwargs(args),
        )
        return 0

    if args.timesnet_command == "evaluate":
        from models.TimesNet.runner import run_evaluate

        run_evaluate(
            args.run_id,
            args.name,
            split=args.split,
            batch_size=args.batch_size,
            n_examples=args.examples,
        )
        return 0

    if args.timesnet_command == "compare":
        from models.TimesNet.runner import run_compare

        run_compare(args.run_id, split=args.split, out_csv=args.out)
        return 0

    if args.timesnet_command == "sweep":
        from models.TimesNet.runner import parse_grid, run_sweep

        run_sweep(
            args.run_id,
            parse_grid(args.grid),
            prefix=args.prefix,
            **_common_train_kwargs(args),
        )
        return 0

    return None
