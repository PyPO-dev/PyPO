from attr import define, field, Factory
from attrs import validators
from src.GUI.utils import inType
import operator


@define
class InputDescription:
    inType : inType
    outputName : str = field(default=None)
    label : str = field(default=None)
    staticValue : str = field(default=None)
    hints : list = Factory(list)
    numFields : int = field(default=1)
    oArray: bool = field(default=False)
    subdict: dict = field(default=None)
    sublist: list = field(default=None)
    hidden: bool = field(default=False)

    def __attrs_post_init__(self):
        if self.outputName and self.label == None:
            self.label = self.outputName.capitalize()
        if len(self.hints) > self.numFields and self.inType.value < 4:
            raise Exception(f"Too many hints! at InputDescription with: inType= {self.inType}, outputName= {self.outputName} \nMake sure to wrap the hints in a list")
        if self.inType.value > 3:
            self.hints = None
            self.numFields = None
        else: 
            self.filloutHints()
        if self.inType is inType.radio:
            # print("radio")
            if self.sublist == None:
                raise Exception("radio without sublist")
        # if not operator.xor(self.subdict == None , self.sublist == None):
        #     raise Exception("Sublist/subdict Error")

    def filloutHints(self):
        while len(self.hints) < self.numFields:
            self.hints.append(str())
        


