

class Batch:
    """
    This class is used to create a dataset of sample points.
    """

    def __init__(self, type_, idx, *args, **kwargs) -> None:
        """
        """
        if type_ == "sample":

            if len(args) != 2:
                raise RuntimeError

            input = args[0]
            conditions = args[1]

            self.input = input[idx]
            self.condition = conditions[idx]

        elif type_ == "data":

            if len(args) != 3:
                raise RuntimeError

            input = args[0]
            output = args[1]
            conditions = args[2]

            self.input = input[idx]
            self.output = output[idx]
            self.condition = conditions[idx]
                
        else:
            raise ValueError("Invalid number of arguments.")