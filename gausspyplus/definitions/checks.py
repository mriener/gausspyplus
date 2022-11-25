import warnings
from typing import List
from gausspyplus.utils.exceptions import AttributeIsNoneException
from gausspyplus.utils.output import format_warning

warnings.showwarning = format_warning


class BaseChecks:
    def set_attribute_if_none(self, name, value, show_warning: bool = False):
        if getattr(self, name) is None:
            setattr(self, name, value)
            if show_warning:
                warnings.warn(
                    f"No value for '{name}' supplied. Setting {name}={value}."
                )

    def raise_exception_if_attribute_is_none(self, attribute, error_message: str = ""):
        if getattr(self, attribute) is None:
            if not error_message:
                error_message = f"You need to specify the '{attribute}' parameter."
            raise AttributeIsNoneException(error_message)

    def raise_exception_if_all_attributes_are_none_or_false(self, attributes: List):
        if not any(getattr(self, attribute) for attribute in attributes):
            raise AttributeIsNoneException(
                f"You need to set at least one of the following parameters: {', '.join(attributes)}"
            )
