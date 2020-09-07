import os

import requests

API_ENDPOINT = os.environ.get("TASKFRAME_API_ENDPOINT", "https://api.taskframe.ai")
API_VERSION = os.environ.get("TASKFRAME_API_VERSION", "v1")
API_URL = f"{API_ENDPOINT}/api/{API_VERSION}"


class ApiError(Exception):
    """API responded with error"""

    def __init__(self, status_code, message):
        super().__init__("<Response [{}]> {}".format(status_code, message))
        self.status_code = status_code


class Client(object):
    def __init__(self):
        self.session = self.create_session()
        self._update_token()
        if os.environ.get("TASKFRAME_SSL_VERIFY") == "False":
            self.session.verify = False

    def create_session(self):
        return requests.Session()

    def get(self, *args, **kwargs):
        return self._send_request("get", *args, **kwargs)

    def put(self, *args, **kwargs):
        return self._send_request("put", *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._send_request("post", *args, **kwargs)

    def _send_request(self, method, url, *args, **kwargs):
        url = f"{API_URL}{url}"
        self._update_token()
        response = getattr(self.session, method)(url, *args, **kwargs)
        if response.status_code >= 400:
            error_message = response.text
            raise ApiError(response.status_code, error_message)
        return response

    def _update_token(self):
        from . import api_key

        self.session.headers.update({"authorization": f"Token {api_key}"})
