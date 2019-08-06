# -*- coding: utf-8 -*-
import math


class Metrics(object):
    """"""

    def __init__(self, model, task="classification"):
        self.model = model
        self.task = task
        self.metric_names = None
        self.metrics_fn = self._infer()

    def evaluate(self, loss, output, target):
        return self.metrics_fn(loss, output, target)

    def _infer(self):
        if self.task == "classification":
            self.topks = (1, 5) if self.model.num_classes >= 5 else (1,)
            self.metric_names = ["top{}".format(topk) for topk in self.topks]
            return self._accuracy
        elif self.task == "language_modeling":
            self.metric_names = ["ppl"]
            return self._ppl
        else:
            raise NotImplementedError

        # some safety check.
        assert self.metric_names is not None

    def _accuracy(self, loss, output, target):
        """Computes the precision@k for the specified values of k"""
        res = []

        if len(self.topks) > 0:
            maxk = max(self.topks)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            for topk in self.topks:
                correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
        else:
            res += [0]
        return res

    def _ppl(self, loss, output, target):
        return [math.exp(loss)]
