class Registry(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.registry = {}

    def register(self, name: str, obj: object) -> None:
        if name in self.registry:
            raise ValueError(f"{self.name} named {name} is already registered.")
        self.registry[name] = obj

    def get(self, name: str) -> object:
        if name not in self.registry:
            raise ValueError(f"{self.name} named {name} is not registered.")
        return self.registry[name]

    def registered(self) -> list[str]:
        return list(self.registry.keys())
