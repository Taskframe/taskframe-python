import requests


class ApiError(Exception):
    """API responded with error"""

    def __init__(self, status_code, message):
        super().__init__("<Response [{}]> {}".format(status_code, message))
        self.status_code = status_code


class Client(object):
    def __init__(self):
        self.session = self.create_session()

    def create_session(self):
        session = requests.Session()
        from . import api_key

        session.headers.update({"authorization": f"Token {api_key}"})
        return session

    def get(self, *args, **kwargs):
        return self._send_request("get", *args, **kwargs)

    def put(self, *args, **kwargs):
        return self._send_request("put", *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._send_request("post", *args, **kwargs)

    def _send_request(self, method, *args, **kwargs):
        response = getattr(self.session, method)(*args, **kwargs)
        if response.status_code >= 400:
            error_message = response.text
            raise ApiError(response.status_code, error_message)
        return response
