import sys

from framework.objects import ivy_instance

ivy_instance.debug = (len(sys.argv) > 1 and
                      sys.argv[1] == "--debug")
ivy_instance.gather("routes")
ivy_instance.run()
