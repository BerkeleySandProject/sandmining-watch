import torch
from os.path import join
from rastervision.pytorch_learner import SemanticSegmentationLearner

class CustomSemanticSegmentationLearner(SemanticSegmentationLearner):
    """
    Rastervisions SemanticSegmentationLearner class provides a lot the functionalities we need.
    In some cases, we want to customize SemanticSegmentationLearner to our needs, we do this here.
    """
    def on_epoch_end(self, curr_epoch, metrics):
        # Default RV saves the model weights to last-model.pth.
        # In the next epoch, RV will overwrite this file.
        # But we want to keep the weights after every epoch
        #
        # This funtion extends the regular on_epoch_end() behaviour.
        super().on_epoch_end(curr_epoch, metrics)
        checkpoint_path = join(self.output_dir_local, f"after-epoch-{curr_epoch}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
