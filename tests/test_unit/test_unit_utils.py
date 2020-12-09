from taskframe.utils import remove_empty_values


class TestUtils:
    def test_remove_empty(self):
        resp = remove_empty_values(
            {"foo": "bar", "nested": {"fizz": False, "buzz": None, "keep": True}}
        )

        assert resp == {"foo": "bar", "nested": {"fizz": False, "keep": True}}
