from enum import Enum
from attr import define, field, Factory
from attrs import validators
import operator

class inType(Enum):
    static = 0
    string = 1
    integers = 2
    floats = 3
    boolean = 4
    dropdown = 5


@define
class InputDescription:
    inType : inType
    outputName : str = field(validator=validators.instance_of(str))
    label : str = field(default=None)
    staticValue : str = field(default=None)
    hints : list = Factory(list)
    numFields : int = 1
    oArray: bool = False
    subdict: dict = field(default=None)
    sublist: list = field(default=None)

    def __attrs_post_init__(self):
        if self.label == None:
            self.label = self.outputName.capitalize()

        if len(self.hints) > self.numFields:
            raise Exception("Too many hints! at InputDescription with: inType= %s, outputName= %s" %(self.inType,self.outputName))
        if self.inType.value == 5:
            self.hints = None
            self.numFields = None
        else: 
            self.filloutHints()
        # if not operator.xor(self.subdict == None , self.sublist == None):
        #     raise Exception("Sublist/subdict Error")

    def filloutHints(self):
        while len(self.hints) < self.numFields:
            self.hints.append(str())
        


