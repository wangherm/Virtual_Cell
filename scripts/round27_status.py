"""Compatibility entrypoint for the persistent agent status command."""
import sys
from round27_agent import main
if __name__=='__main__':
    sys.argv.insert(1,'status')
    main()
