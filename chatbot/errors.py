class UserFacingError(Exception):
    """Expected input error: the message is safe to display to the user"""

    pass


class AuthenticationRequiredError(Exception):
    """Raise error when user is not authenticated or token is not valid"""

    pass


class FormRegisterException(Exception):
    """Raise error when the register form validation fails"""

    pass


class FormLoginException(Exception):
    """Raise error when the login form validation fails"""

    pass


class StorageException(Exception):
    """Raise error when storage operation fails"""

    pass
