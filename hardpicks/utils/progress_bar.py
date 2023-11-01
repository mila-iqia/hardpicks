from pytorch_lightning.callbacks import RichProgressBar


class CustomRichProgressBar(RichProgressBar):
    """A custom progress bar, based on Rich."""
    def get_metrics(self, trainer, model):
        """Get the metrics to show on the progress bar."""
        # Get rid of that useless version number.
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
