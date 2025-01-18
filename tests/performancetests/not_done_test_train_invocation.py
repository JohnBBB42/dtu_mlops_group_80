from energy.train import main


def test_main_runs_without_error(monkeypatch):
    # Monkey-patch trainer to avoid actual training
    import pytorch_lightning as pl

    class DummyTrainer:
        def fit(self, *args, **kwargs):
            pass

        def test(self, *args, **kwargs):
            pass

    monkeypatch.setattr(pl, "Trainer", lambda **kwargs: DummyTrainer())

    # Run main, which should call our dummy trainer without error
    main()
