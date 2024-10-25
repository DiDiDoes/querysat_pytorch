class ValueMeter(object):
    def __init__(self, name: str, fmt: str = "{}"):
        self.name = name
        self.fmt = fmt
        self.fmtstr = "{:s} " + fmt + " (" + fmt + ")"
        self.reset()

    def reset(self):
        self.value = 0
        self.result = 0

    def update(self, value, n: int = 1):
        self.value = value

    def report_str(self):
        return self.fmtstr.format(self.name, self.value, self.result)

    def summary_str(self):
        fmtstr = "{:s} " + self.fmt
        return fmtstr.format(self.name, self.result)


class AverageMeter(ValueMeter):
    def reset(self):
        super().reset()
        self.sum = 0
        self.count = 0

    def update(self, value, n: int = 1):
        super().update(value, n)
        self.sum += value * n
        self.count += n
        self.result = self.sum / self.count


class SumMeter(ValueMeter):
    def update(self, value, n: int = 1):
        super().update(value, n)
        self.result += value * n


class MaxMeter(ValueMeter):
    def reset(self):
        super().reset()
        self.result = float("-inf")

    def update(self, value, n: int = 1):
        super().update(value, n)
        self.result = max(self.result, value)


class ProgressMeter(object):
    def __init__(
            self,
            name: str,
            meters: list[ValueMeter],
            n: int | None = None,
        ):
        self.name = name
        self.meters = meters

        if n is None:
            self.fmtstr = "[{:d}]"
        else:
            n_digit = len(str(int(n)))
            self.fmtstr = "{:" + str(n_digit) + "d}"
            self.fmtstr = "[" + self.fmtstr + "/" + self.fmtstr.format(n) + "]"

    def title_str(self):
        return '='*20 + " " + self.name + " " + '='*20

    def report_str(self, count):
        progress = [self.name + self.fmtstr.format(count)]
        progress += [meter.report_str() for meter in self.meters]
        return ' | '.join(progress)

    def summary_str(self):
        progress = [self.name]
        progress += [meter.summary_str() for meter in self.meters]
        return ' | '.join(progress)
