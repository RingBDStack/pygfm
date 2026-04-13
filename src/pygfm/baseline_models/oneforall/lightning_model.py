from pygfm.baseline_models.oneforall.gp.lightning.module_template import BaseTemplate


class GraphPredLightning(BaseTemplate):
    def forward(self, batch):
        return self.model(batch)
