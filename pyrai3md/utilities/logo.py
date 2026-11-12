"""
======================================================================
This module provides the PyRAI3MD startup logo and version information.
It displays the program name, ASCII art logo, and version number
when the program is launched.

Jul 28 2026
By Jingbai Li

Hoffmann Institute of Advanced Materials
Shenzhen Polytechnic University
======================================================================
"""

__version__ = '3.0'


def print_logo():
    """Return the PyRAI3MD startup logo string.

    Returns
    -------
    str
        The formatted logo string with version number.
    """
    logo = """
  -------------------------------------------------------------------
                              /\\
   |\\    /|                  /++\\
   ||\\  /||                 /++++\\
   || \\/ || ||             /++++++\\
   ||    || ||            /PyRAI3MD\\
   ||    || ||           /++++++++++\\                     __
            ||          /++++++++++++\\     |\\ |  /\\  |\\/| | \\
            ||__ __    *==============*    | \\| /--\\ |  | |_/


                      Power your Research 
                            with
      Artificial Intelligence-ready Advanced Infrastructure
                            for 
                  Ab Initio Molecular Dynamics

                      Author: Jingbai Li

            Hoffmann Institute of Advanced Materials
                  Shenzhen Polytechnic University

                             2026

                          version:   %s
  -------------------------------------------------------------------
""" % __version__

    return logo
