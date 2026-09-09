import unittest

from mlx_edge.replace import LiteralReplacer, parse_replace_rules


class ReplaceTests(unittest.TestCase):
    def test_parse_space_colon_pairs(self):
        rules = parse_replace_rules("{foo : bar}\n{baz : qux}")
        self.assertEqual(rules, [("foo", "bar"), ("baz", "qux")])

    def test_parse_empty_replace(self):
        self.assertEqual(parse_replace_rules("{]<]minimax[>[ : }"), [("]<]minimax[>[", "")])

    def test_parse_tight_colon(self):
        self.assertEqual(parse_replace_rules("{foo:bar}"), [("foo", "bar")])

    def test_stream_across_chunks(self):
        rep = LiteralReplacer([("Hello", "Hi")])
        self.assertEqual(rep.push("He"), "")
        self.assertEqual(rep.push("llo world"), "Hi world")
        self.assertEqual(rep.flush(), "")

    def test_delete_token(self):
        rep = LiteralReplacer([("<think>", "")])
        self.assertEqual(rep.push("<th"), "")
        self.assertEqual(rep.push("ink>rest"), "rest")

    def test_first_longest_match(self):
        rep = LiteralReplacer([("ab", "X"), ("abc", "Y")])
        self.assertEqual(rep.push("abc!"), "Y!")

    def test_flush_partial(self):
        rep = LiteralReplacer([("Hello", "Hi")])
        self.assertEqual(rep.push("Hel"), "")
        self.assertEqual(rep.flush(), "Hel")


if __name__ == "__main__":
    unittest.main()
