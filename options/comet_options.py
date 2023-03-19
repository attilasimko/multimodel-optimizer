
class CometOptions():
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    def initialize(self, parser):
        parser.add_argument("--log_comet", default="False")
        return parser


