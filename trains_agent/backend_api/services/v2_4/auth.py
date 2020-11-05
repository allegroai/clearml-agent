"""
auth service

This service provides authentication management and authorization
validation for the entire system.
"""
import six
import types
from datetime import datetime
import enum

from dateutil.parser import parse as parse_datetime

from ....backend_api.session import Request, BatchRequest, Response, DataModel, NonStrictDataModel, CompoundRequest, schema_property, StringEnum


class Credentials(NonStrictDataModel):
    """
    :param access_key: Credentials access key
    :type access_key: str
    :param secret_key: Credentials secret key
    :type secret_key: str
    """
    _schema = {
        'properties': {
            'access_key': {
                'description': 'Credentials access key',
                'type': ['string', 'null'],
            },
            'secret_key': {
                'description': 'Credentials secret key',
                'type': ['string', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, access_key=None, secret_key=None, **kwargs):
        """
        Initialize a new access token.

        Args:
            self: (todo): write your description
            access_key: (str): write your description
            secret_key: (str): write your description
        """
        super(Credentials, self).__init__(**kwargs)
        self.access_key = access_key
        self.secret_key = secret_key

    @schema_property('access_key')
    def access_key(self):
        """
        The access key.

        Args:
            self: (todo): write your description
        """
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        """
        Returns the access key.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_access_key = None
            return

        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value

    @schema_property('secret_key')
    def secret_key(self):
        """
        The secret key.

        Args:
            self: (todo): write your description
        """
        return self._property_secret_key

    @secret_key.setter
    def secret_key(self, value):
        """
        The secret key.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_secret_key = None
            return

        self.assert_isinstance(value, "secret_key", six.string_types)
        self._property_secret_key = value


class CredentialKey(NonStrictDataModel):
    """
    :param access_key:
    :type access_key: str
    :param last_used:
    :type last_used: datetime.datetime
    :param last_used_from:
    :type last_used_from: str
    """
    _schema = {
        'properties': {
            'access_key': {'description': '', 'type': ['string', 'null']},
            'last_used': {
                'description': '',
                'format': 'date-time',
                'type': ['string', 'null'],
            },
            'last_used_from': {'description': '', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, access_key=None, last_used=None, last_used_from=None, **kwargs):
        """
        Initialize access_key. : param access_key : param last_used : : return :

        Args:
            self: (todo): write your description
            access_key: (str): write your description
            last_used: (bool): write your description
            last_used_from: (str): write your description
        """
        super(CredentialKey, self).__init__(**kwargs)
        self.access_key = access_key
        self.last_used = last_used
        self.last_used_from = last_used_from

    @schema_property('access_key')
    def access_key(self):
        """
        The access key.

        Args:
            self: (todo): write your description
        """
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        """
        Returns the access key.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_access_key = None
            return

        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value

    @schema_property('last_used')
    def last_used(self):
        """
        Returns the last available property.

        Args:
            self: (todo): write your description
        """
        return self._property_last_used

    @last_used.setter
    def last_used(self, value):
        """
        Returns the last none if none.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_last_used = None
            return

        self.assert_isinstance(value, "last_used", six.string_types + (datetime,))
        if not isinstance(value, datetime):
            value = parse_datetime(value)
        self._property_last_used = value

    @schema_property('last_used_from')
    def last_used_from(self):
        """
        Returns the last available property.

        Args:
            self: (todo): write your description
        """
        return self._property_last_used_from

    @last_used_from.setter
    def last_used_from(self, value):
        """
        Return the last property.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_last_used_from = None
            return

        self.assert_isinstance(value, "last_used_from", six.string_types)
        self._property_last_used_from = value




class CreateCredentialsRequest(Request):
    """
    Creates a new set of credentials for the authenticated user.
                            New key/secret is returned.
                            Note: Secret will never be returned in any other API call.
                            If a secret is lost or compromised, the key should be revoked
                            and a new set of credentials can be created.

    """

    _service = "auth"
    _action = "create_credentials"
    _version = "2.4"
    _schema = {
        'additionalProperties': False,
        'definitions': {},
        'properties': {},
        'type': 'object',
    }


class CreateCredentialsResponse(Response):
    """
    Response of auth.create_credentials endpoint.

    :param credentials: Created credentials
    :type credentials: Credentials
    """
    _service = "auth"
    _action = "create_credentials"
    _version = "2.4"

    _schema = {
        'definitions': {
            'credentials': {
                'properties': {
                    'access_key': {
                        'description': 'Credentials access key',
                        'type': ['string', 'null'],
                    },
                    'secret_key': {
                        'description': 'Credentials secret key',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'credentials': {
                'description': 'Created credentials',
                'oneOf': [{'$ref': '#/definitions/credentials'}, {'type': 'null'}],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, credentials=None, **kwargs):
        """
        Initialize credentials.

        Args:
            self: (todo): write your description
            credentials: (todo): write your description
        """
        super(CreateCredentialsResponse, self).__init__(**kwargs)
        self.credentials = credentials

    @schema_property('credentials')
    def credentials(self):
        """
        : class credentials.

        Args:
            self: (todo): write your description
        """
        return self._property_credentials

    @credentials.setter
    def credentials(self, value):
        """
        Retrieves credentials. credentials.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_credentials = None
            return
        if isinstance(value, dict):
            value = Credentials.from_dict(value)
        else:
            self.assert_isinstance(value, "credentials", Credentials)
        self._property_credentials = value




class EditUserRequest(Request):
    """
     Edit a users' auth data properties

    :param user: User ID
    :type user: str
    :param role: The new user's role within the company
    :type role: str
    """

    _service = "auth"
    _action = "edit_user"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'role': {
                'description': "The new user's role within the company",
                'enum': ['admin', 'superuser', 'user', 'annotator'],
                'type': ['string', 'null'],
            },
            'user': {'description': 'User ID', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, user=None, role=None, **kwargs):
        """
        Initialize the user

        Args:
            self: (todo): write your description
            user: (str): write your description
            role: (str): write your description
        """
        super(EditUserRequest, self).__init__(**kwargs)
        self.user = user
        self.role = role

    @schema_property('user')
    def user(self):
        """
        Return the user object.

        Args:
            self: (todo): write your description
        """
        return self._property_user

    @user.setter
    def user(self, value):
        """
        The user property.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_user = None
            return

        self.assert_isinstance(value, "user", six.string_types)
        self._property_user = value

    @schema_property('role')
    def role(self):
        """
        Return the role.

        Args:
            self: (todo): write your description
        """
        return self._property_role

    @role.setter
    def role(self, value):
        """
        The role of the role

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_role = None
            return

        self.assert_isinstance(value, "role", six.string_types)
        self._property_role = value


class EditUserResponse(Response):
    """
    Response of auth.edit_user endpoint.

    :param updated: Number of users updated (0 or 1)
    :type updated: float
    :param fields: Updated fields names and values
    :type fields: dict
    """
    _service = "auth"
    _action = "edit_user"
    _version = "2.4"

    _schema = {
        'definitions': {},
        'properties': {
            'fields': {
                'additionalProperties': True,
                'description': 'Updated fields names and values',
                'type': ['object', 'null'],
            },
            'updated': {
                'description': 'Number of users updated (0 or 1)',
                'enum': [0, 1],
                'type': ['number', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, updated=None, fields=None, **kwargs):
        """
        Set the fields of the fields. fields.

        Args:
            self: (todo): write your description
            updated: (todo): write your description
            fields: (dict): write your description
        """
        super(EditUserResponse, self).__init__(**kwargs)
        self.updated = updated
        self.fields = fields

    @schema_property('updated')
    def updated(self):
        """
        Retrieves the updated object.

        Args:
            self: (todo): write your description
        """
        return self._property_updated

    @updated.setter
    def updated(self, value):
        """
        Returns the updated update value

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_updated = None
            return

        self.assert_isinstance(value, "updated", six.integer_types + (float,))
        self._property_updated = value

    @schema_property('fields')
    def fields(self):
        """
        : return : class : class :.

        Args:
            self: (todo): write your description
        """
        return self._property_fields

    @fields.setter
    def fields(self, value):
        """
        Return a dictionary of fields.

        Args:
            self: (todo): write your description
            value: (str): write your description
        """
        if value is None:
            self._property_fields = None
            return

        self.assert_isinstance(value, "fields", (dict,))
        self._property_fields = value


class GetCredentialsRequest(Request):
    """
    Returns all existing credential keys for the authenticated user.
            Note: Only credential keys are returned.

    """

    _service = "auth"
    _action = "get_credentials"
    _version = "2.4"
    _schema = {
        'additionalProperties': False,
        'definitions': {},
        'properties': {},
        'type': 'object',
    }


class GetCredentialsResponse(Response):
    """
    Response of auth.get_credentials endpoint.

    :param credentials: List of credentials, each with an empty secret field.
    :type credentials: Sequence[CredentialKey]
    """
    _service = "auth"
    _action = "get_credentials"
    _version = "2.4"

    _schema = {
        'definitions': {
            'credential_key': {
                'properties': {
                    'access_key': {'description': '', 'type': ['string', 'null']},
                    'last_used': {
                        'description': '',
                        'format': 'date-time',
                        'type': ['string', 'null'],
                    },
                    'last_used_from': {
                        'description': '',
                        'type': ['string', 'null'],
                    },
                },
                'type': 'object',
            },
        },
        'properties': {
            'credentials': {
                'description': 'List of credentials, each with an empty secret field.',
                'items': {'$ref': '#/definitions/credential_key'},
                'type': ['array', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, credentials=None, **kwargs):
        """
        Initialize authentication.

        Args:
            self: (todo): write your description
            credentials: (todo): write your description
        """
        super(GetCredentialsResponse, self).__init__(**kwargs)
        self.credentials = credentials

    @schema_property('credentials')
    def credentials(self):
        """
        : class credentials.

        Args:
            self: (todo): write your description
        """
        return self._property_credentials

    @credentials.setter
    def credentials(self, value):
        """
        Gets / sets the credential credentials.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_credentials = None
            return

        self.assert_isinstance(value, "credentials", (list, tuple))
        if any(isinstance(v, dict) for v in value):
            value = [CredentialKey.from_dict(v) if isinstance(v, dict) else v for v in value]
        else:
            self.assert_isinstance(value, "credentials", CredentialKey, is_array=True)
        self._property_credentials = value




class LoginRequest(Request):
    """
    Get a token based on supplied credentials (key/secret).
            Intended for use by users with key/secret credentials that wish to obtain a token
            for use with other services. Token will be limited by the same permissions that
            exist for the credentials used in this call.

    :param expiration_sec: Requested token expiration time in seconds. Not
        guaranteed,  might be overridden by the service
    :type expiration_sec: int
    """

    _service = "auth"
    _action = "login"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'expiration_sec': {
                'description': 'Requested token expiration time in seconds. \n                        Not guaranteed,  might be overridden by the service',
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, expiration_sec=None, **kwargs):
        """
        Initialize the expiration.

        Args:
            self: (todo): write your description
            expiration_sec: (todo): write your description
        """
        super(LoginRequest, self).__init__(**kwargs)
        self.expiration_sec = expiration_sec

    @schema_property('expiration_sec')
    def expiration_sec(self):
        """
        Returns the expiration expiration expiration_expiration.

        Args:
            self: (todo): write your description
        """
        return self._property_expiration_sec

    @expiration_sec.setter
    def expiration_sec(self, value):
        """
        The expiration property.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_expiration_sec = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "expiration_sec", six.integer_types)
        self._property_expiration_sec = value


class LoginResponse(Response):
    """
    Response of auth.login endpoint.

    :param token: Token string
    :type token: str
    """
    _service = "auth"
    _action = "login"
    _version = "2.4"

    _schema = {
        'definitions': {},
        'properties': {
            'token': {'description': 'Token string', 'type': ['string', 'null']},
        },
        'type': 'object',
    }
    def __init__(
            self, token=None, **kwargs):
        """
        Initialize the token.

        Args:
            self: (todo): write your description
            token: (str): write your description
        """
        super(LoginResponse, self).__init__(**kwargs)
        self.token = token

    @schema_property('token')
    def token(self):
        """
        Returns the token as a dict.

        Args:
            self: (todo): write your description
        """
        return self._property_token

    @token.setter
    def token(self, value):
        """
        The token property.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_token = None
            return

        self.assert_isinstance(value, "token", six.string_types)
        self._property_token = value


class LogoutRequest(Request):
    """
    Removes the authentication cookie from the current session

    """

    _service = "auth"
    _action = "logout"
    _version = "2.4"
    _schema = {'additionalProperties': False, 'definitions': {}, 'type': 'object'}


class LogoutResponse(Response):
    """
    Response of auth.logout endpoint.

    """
    _service = "auth"
    _action = "logout"
    _version = "2.4"

    _schema = {'additionalProperties': False, 'definitions': {}, 'type': 'object'}


class RevokeCredentialsRequest(Request):
    """
    Revokes (and deletes) a set (key, secret) of credentials for
            the authenticated user.

    :param access_key: Credentials key
    :type access_key: str
    """

    _service = "auth"
    _action = "revoke_credentials"
    _version = "2.4"
    _schema = {
        'definitions': {},
        'properties': {
            'access_key': {
                'description': 'Credentials key',
                'type': ['string', 'null'],
            },
        },
        'required': ['key_id'],
        'type': 'object',
    }
    def __init__(
            self, access_key=None, **kwargs):
        """
        Initialize an access token.

        Args:
            self: (todo): write your description
            access_key: (str): write your description
        """
        super(RevokeCredentialsRequest, self).__init__(**kwargs)
        self.access_key = access_key

    @schema_property('access_key')
    def access_key(self):
        """
        The access key.

        Args:
            self: (todo): write your description
        """
        return self._property_access_key

    @access_key.setter
    def access_key(self, value):
        """
        Returns the access key.

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_access_key = None
            return

        self.assert_isinstance(value, "access_key", six.string_types)
        self._property_access_key = value


class RevokeCredentialsResponse(Response):
    """
    Response of auth.revoke_credentials endpoint.

    :param revoked: Number of credentials revoked
    :type revoked: int
    """
    _service = "auth"
    _action = "revoke_credentials"
    _version = "2.4"

    _schema = {
        'definitions': {},
        'properties': {
            'revoked': {
                'description': 'Number of credentials revoked',
                'enum': [0, 1],
                'type': ['integer', 'null'],
            },
        },
        'type': 'object',
    }
    def __init__(
            self, revoked=None, **kwargs):
        """
        Initialize the credentials.

        Args:
            self: (todo): write your description
            revoked: (str): write your description
        """
        super(RevokeCredentialsResponse, self).__init__(**kwargs)
        self.revoked = revoked

    @schema_property('revoked')
    def revoked(self):
        """
        Access the : meth : ~zhitem.

        Args:
            self: (todo): write your description
        """
        return self._property_revoked

    @revoked.setter
    def revoked(self, value):
        """
        Gets / sets the revision property

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        if value is None:
            self._property_revoked = None
            return
        if isinstance(value, float) and value.is_integer():
            value = int(value)

        self.assert_isinstance(value, "revoked", six.integer_types)
        self._property_revoked = value




response_mapping = {
    LoginRequest: LoginResponse,
    LogoutRequest: LogoutResponse,
    CreateCredentialsRequest: CreateCredentialsResponse,
    GetCredentialsRequest: GetCredentialsResponse,
    RevokeCredentialsRequest: RevokeCredentialsResponse,
    EditUserRequest: EditUserResponse,
}
