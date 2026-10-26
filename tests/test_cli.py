import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from private_research_assistant import UserFacingError
from private_research_assistant.cli import main


class CliTests(unittest.TestCase):
    def test_remaining_commands_call_their_handler(self):
        for command, arguments in (
            ("setup-check", []),
            ("crawl", []),
            ("index", []),
            ("extract", []),
            ("ask", ["Which positions are funded?"]),
            ("inspect", ["opportunities"]),
        ):
            with self.subTest(command=command):
                handler_name = "_cmd_" + command.replace("-", "_")
                with patch(f"private_research_assistant.cli.{handler_name}", return_value=0) as handler:
                    self.assertEqual(main([command, *arguments]), 0)
                    handler.assert_called_once()
                    self.assertEqual(handler.call_args.args[0].command, command)

    def test_removed_commands_are_rejected(self):
        for arguments in (["eval"], ["eval", "--live"], ["demo"], ["status"]):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_no_command_shows_help(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([]), 0)
        self.assertIn("setup-check,crawl,index,extract,ask,inspect", output.getvalue())

    def test_expected_errors_and_interrupts_have_exit_codes(self):
        for error, code in (
            (UserFacingError("missing data"), 1),
            (FileNotFoundError("missing file"), 1),
            (KeyboardInterrupt(), 130),
        ):
            with self.subTest(error=type(error).__name__), redirect_stderr(io.StringIO()):
                with patch("private_research_assistant.cli.build_index", side_effect=error):
                    self.assertEqual(main(["index"]), code)
