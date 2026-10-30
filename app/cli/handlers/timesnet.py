"""Dispatch handlers for TimesNet commands."""

from models.TimesNet import config as tc

from app.cli.helpers import split_locations


def dispatch(args, parser) -> int | None:
    if args.command != "timesnet":
        return None

    if args.timesnet_command == "prepare":
        from models.TimesNet.runner import run_prepare as timesnet_prepare

        locations = split_locations(args.locations)
        if not locations:
            parser.error("--locations must not be empty")
        run_id = args.run_id or tc.default_run_id()
        timesnet_prepare(
            run_id=run_id,
            locations=locations,
            date_from=args.date_from,
            date_to=args.date_to,
            freq=args.freq,
            feature_table=args.feature_table,
        )
        return 0

    if args.timesnet_command == "prepare-class":
        from models.TimesNet.runner import run_prepare_class as timesnet_prepare_class

        locations = split_locations(args.locations)
        if not locations:
            parser.error("--locations must not be empty")
        run_id = args.run_id or tc.default_run_id()
        timesnet_prepare_class(
            run_id=run_id,
            locations=locations,
            date_from=args.date_from,
            date_to=args.date_to,
            freq=args.freq,
            min_nonzero_hours_per_series=args.min_nonzero_hours_per_series,
        )
        return 0

    if args.timesnet_command == "train":
        from models.TimesNet.runner import run_train as timesnet_train

        pin_memory = args.pin_memory
        if args.no_pin_memory:
            pin_memory = False
        use_rmse_loss = args.use_rmse_loss
        if args.no_rmse_loss:
            use_rmse_loss = False

        timesnet_train(
            run_id=args.run_id,
            window=args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            d_model=args.d_model,
            d_ff=args.d_ff,
            e_layers=args.e_layers,
            top_k=args.top_k,
            num_kernels=args.num_kernels,
            dropout=args.dropout,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=not args.no_persistent_workers,
            use_amp=args.use_amp,
            use_rmse_loss=use_rmse_loss,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            log_every=args.log_every,
            oom_retry_enabled=not args.no_oom_retry,
            oom_retry_min_batch=args.oom_retry_min_batch,
            lr_scheduler_enabled=not args.no_lr_scheduler,
            lr_scheduler_patience=args.lr_scheduler_patience,
            lr_scheduler_factor=args.lr_scheduler_factor,
        )
        return 0

    if args.timesnet_command == "eval":
        from models.TimesNet.runner import run_eval as timesnet_eval

        max_records = args.max_records or None
        plot_top = args.plot_top if args.plot_top > 0 else None
        timesnet_eval(
            run_id=args.run_id,
            threshold_profile=args.threshold_profile,
            threshold_method=args.threshold_method,
            threshold=args.threshold,
            quantile=args.quantile,
            combined_robust_z=args.combined_robust_z,
            domain_filter=not args.no_domain_filter,
            score_smoothing_span=args.score_smoothing_span,
            eval_split=args.eval_split,
            date_from=args.date_from,
            date_to=args.date_to,
            max_records=max_records,
            plot_top=plot_top,
            include_train_in_score_plot=args.include_train_in_score_plot,
            zoom_to_eval=not args.no_zoom_to_eval,
            include_full_span_plot=not args.no_full_span_plot,
            gt_eval=args.gt_eval,
            gt_csv=args.gt_csv,
        )
        return 0

    if args.timesnet_command == "gt-eval":
        from models.TimesNet.runner import run_gt_eval as timesnet_gt_eval

        timesnet_gt_eval(
            run_id=args.run_id,
            eval_split=args.eval_split,
            gt_csv=args.gt_csv,
        )
        return 0

    return None
