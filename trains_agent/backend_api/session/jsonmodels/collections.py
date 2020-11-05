

class ModelCollection(list):

    """`ModelCollection` is list which validates stored values.

    Validation is made with use of field passed to `__init__` at each point,
    when new value is assigned.

    """

    def __init__(self, field):
        """
        Initialize the field

        Args:
            self: (todo): write your description
            field: (todo): write your description
        """
        self.field = field

    def append(self, value):
        """
        Add a single field to the list.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self.field.validate_single_value(value)
        super(ModelCollection, self).append(value)

    def __setitem__(self, key, value):
        """
        Set the value of the field.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (str): write your description
        """
        self.field.validate_single_value(value)
        super(ModelCollection, self).__setitem__(key, value)
