from attrs import define, field, Factory
from src.GUI.utils import inType, InputDescriptionError

##
# @file 
# Defines an attr dataclass used to store the data of parameter forms.
# 

## 
# @class
# This data is fed into a form generator to be interpreted and to generate the form. 
# The data for a form consist of a list of InputDescription objects. If these objects are of a dynamic type 
# (e.g. dynamicDropdown or dynamicRadio) then they may contain a Dictionary (subDict) storing at each of 
# the values a list of inputDescriptions. These lists are fed into new form generators allowing for nested forms.   
@define
class InputDescription:
    ## 
    # Determines the type of the input.
    # 
    # @see inType
    inType : inType= field()
    
    ##
    # String representing the key in the output dictionary.
    outputName = field(default=None)
    
    ##
    # Label for the form, if not set then the outputName will be used.
    label = field(default=None)
    
    ##
    # Value to be output if inType is static. In such case the user is not able to interact with the input 
    # field. This value represents the predetermined output.
    staticValue = field(default=None)
    
    ##
    # List of values, used if inType is a vector type. Can be used as hints or as prefilled values if 'prefill' is True.
    # Also used in 'SimpleRadio', 'SimpleDropdown' and 'ElementSelectionWidget' to display hints instead of options. In such
    # cases options are still used to read the form, but this way you can make a difference between what the user sees and 
    # how the choice is interpreted.
    hints = field(default=Factory(list))
    
    ##
    # Number of input fields, used if inType is a vector type. 
    numFields = field(default=None)
    
    ##
    # Determines wether the output will be converted to a numpy array, used if inType is a vector type.
    oArray = field(default=None)
    
    ##
    # Dictionary containing strings as keys and lists of inputDescriptions as values. Used to nest
    # forms if inType is a dynamic type.
    subDict = field(default=None)
    
    ##
    # List of options used in 'SimpleRadio', 'SimpleDropdown' and 'ElementSelectionWidget' to provide options.
    #
    # @see hints 
    options = field(default=None)
    
    ##
    # Hides the input.
    # Used in case of a static input to hide the input.
    hidden = field(default=False)
    
    ##
    # used by all inputs to provide a tooltip.
    toolTip = field(default=None)
    
    ##
    # If true a vector type input will use the hints as prefilled values.
    # If true a checkbox input will be checked by default.
    prefill = field(default=False)

    ##
    # General validation of inputDescriptions.
    def __attrs_post_init__(self):
        vectorTypes = [inType.vectorFloats, inType.vectorIntegers, inType.vectorStrings]
        selectionTypes = [inType.radio, inType.dropdown]
        dynamicTypes = [inType.dynamicRadio, inType.dynamicDropdown]
        # Validation for inType
        if type(self.inType) != inType:
            raise InputDescriptionError("Cannot define an input description without an inType.")
        
        # Validation for outputName
        if self.outputName is None:
            if self.inType != inType.static:
                raise InputDescriptionError("outputName required.")
            if self.label is None:
                self.label = "Unnamed"

        # Derivation for label
        if self.outputName and self.label is None:
            self.label = self.outputName.capitalize()

        # Validation for staticValue
        if self.staticValue and self.inType != inType.static:
            raise InputDescriptionError("Cannot set static value for input of non static type.")
                
        # Validation for numFields
        if self.numFields is None:
            if self.inType in vectorTypes:
                self.numFields = 1
        if self.numFields is not None and self.inType not in vectorTypes:
            raise InputDescriptionError(f"Cannot set numFields for InputDescription of inType:{self.inType}.")

        # Validation for hints
        if self.hints: 
            if self.inType not in vectorTypes+selectionTypes:
                raise InputDescriptionError(f"Cannot set hints for InputDescription of inType:{self.inType}.")
            if self.inType in vectorTypes and len(self.hints) != self.numFields:
                raise InputDescriptionError("Number of hints does not match numFields.")
            if self.inType in selectionTypes and len(self.hints) != len(self.options):
                raise InputDescriptionError("Number of hints does not match number of options.")
        
        # Validation for oArray
        if self.oArray and not self.inType in [inType.vectorFloats, inType.vectorIntegers]:
            raise InputDescriptionError(f"Cannot set oArray for InputDescription of inType:{self.inType}.")

        # Validation for subDict
        if self.subDict:
            if self.inType not in dynamicTypes:
                raise InputDescriptionError(f"Cannot set subDict for InputDescription of inType:{self.inType}.")
            if len(self.subDict) < 2:
                raise InputDescriptionError(f"The subDict of a dynamic type inputDescription should have at least two entries.")
            for key, value in self.subDict.items():
                if type(key) != str:
                    raise InputDescriptionError(f"Unexpected type:{type(key)} Keys of subDict must be of type str.")
                if type(value) != list:
                    raise InputDescriptionError(f"Unexpected type:{type(value)} Values of subDict must be of type list.")
                for i in value:
                    if type(i) != InputDescription:
                        raise InputDescriptionError(f"Unexpected type:{type(i)} Values of subDict must be lists of InputDescription objects.")
                        
        # Validation for hidden
        if self.hidden:
            if self.inType != inType.static:
                raise InputDescriptionError(f"InputDescription of type {self.inType} cannot be hidden.")
            if self.outputName is None:
                raise InputDescriptionError(f"Hidden InputDescription that has no outputName does nothing and should be removed")

        # Validation for prefill
        if self.prefill:
            if self.inType in vectorTypes and self.hints is None:
                raise InputDescriptionError(f"Hints required to prefill a vector type InputDescription")

            if self.inType not in vectorTypes + [inType.checkbox]:
                raise InputDescriptionError(f"Cannot set prefill for InputDescription of inType:{self.inType}.")

        # Validation for options
        if self.options:
            if self.inType not in selectionTypes + [inType.elementSelector]:
                raise InputDescriptionError(f"Cannot set options for InputDescription of inType:{self.inType}.")
