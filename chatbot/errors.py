class UserFacingError(Exception):
    """Expected input error: the message is safe to display to the user"""

    pass


class AuthenticationRequiredError(Exception):
    """Raise error when user is not authenticated or token is not valid"""

    pass
