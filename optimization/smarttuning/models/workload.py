from datetime import datetime, timedelta


class Workload:
    def __init__(self, name: str, min_lifespan_lenght_s: int = 0, data=None):
        self.__name: str = name
        self.__creation_time: datetime = datetime.utcnow()
        self.__min_lifespan_lenght_s: int = min_lifespan_lenght_s
        self.__ctx = None
        self.__data = data

    def __eq__(self, other):
        return other is not None and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return f'Workload(name={self.name}, data={self.data})'

    def serialize(self):
        return {
            'name': self.name,
            'data': self.data,
            'creation_time': self.creation_time.isoformat(),
            'min_lifespan_valid': self.__min_lifespan_lenght_s,
            'lifespan': self.lifespan().total_seconds(),
            'valid': self.is_valid()
        }

    @property
    def name(self) -> str:
        return self.__name

    @property
    def creation_time(self) -> datetime:
        return self.__creation_time

    @property
    def ctx(self):
        # smarttuningcontext
        return self.__ctx

    @ctx.setter
    def ctx(self, value):
        self.__ctx = value

    @property
    def data(self):
        return self.__data

    def reset(self):
        self.__creation_time = datetime.utcnow()

    def lifespan(self) -> timedelta:
        return datetime.utcnow() - self.creation_time

    def is_valid(self) -> bool:
        return self.lifespan().total_seconds() >= self.__min_lifespan_lenght_s


class EmptyWorkload(Workload):
    def __init__(self):
        super(EmptyWorkload, self).__init__('INVALID', -1)


empty_workload = EmptyWorkload()
