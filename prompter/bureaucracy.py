

class NextStep:
    paperwork: Optional[BaseModel]
    step: Bureaucrat


class Bureaucrat:
    processes: List[NextStep]

    def shuffle_papers(self, *args, **kwargs):
        raise NotImplementedError()


class SearchProperties(Bureaucrat):
    processes = [
        NextStep(paperwork=None, step=MaveAgent),
        NextStep(paperwork=PropertySchema, step=MaveAgent),
    ]

    def shuffle_papers(self, *args, **kwargs):
        raise NotImplementedError()


class MaveAgent(Bureaucrat):
    processes = [
        NextStep(paperwork=PropertySearchSchema,
                 step=SearchPropertiesAgent),
    ]

    def shuffle_papers(self, *args, **kwargs):
        raise NotImplementedError()
