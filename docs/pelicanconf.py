# -*- coding: utf-8 -*- #
#!/usr/bin/env python
"""
Simple Pelican configuration file.

Copy this file to <your‑repo>/docs/pelicanconf.py (or wherever you keep
your documentation source) and adjust the values marked with **TODO**.
"""

############################
# Basic site information   #
############################

# The title that appears in the generated HTML <title> tag and in the header.
AUTHOR = 'Laurent Perrinet'                     # TODO: put your name or org
SITENAME = 'My Project Documentation'   # TODO: project/site title
SITEURL = 'https://retinoto-py.readthedocs.io'                             # Empty for local builds; set to the live URL for production

# Path where your markdown/reStructuredText content lives.
# Usually “content” is the default for Pelican, but you can point it
# anywhere you keep the .md/.rst files.
PATH = '.'                         # TODO: change if you store docs elsewhere

# Time‑zone and default language – keep the defaults unless you need something else.
TIMEZONE = 'CET'
DEFAULT_LANG = 'en'

############################
# Content handling         #
############################

# List of file extensions Pelican will treat as articles/pages.
# Keep the defaults unless you have a custom extension.
MARKUP = ('md', 'ipynb')


# If you want a blog‑style “articles” section, enable this.
# If you only have static pages, you can leave ARTICLE_PATHS empty.
# ARTICLE_PATHS = ['articles']   # put article files here (optional)
PAGE_PATHS = ['docs']         # put pure pages here (e.g., “about”, “install”)

# What URL format you want for articles/pages.
# The simple form below works well for documentation sites.
ARTICLE_URL = 'posts/{slug}/'
ARTICLE_SAVE_AS = 'posts/{slug}/index.html'
PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'

############################
# Theme & appearance       #
############################

# Pelican ships with a few built‑in themes.  “notmyidea” works out of the box.
# You can replace it with any theme you like (install it via pip or clone it).
THEME = 'notmyidea'                     # TODO: replace with your favourite theme path

# Optional: static files (CSS, images, etc.) that live next to this config.
STATIC_PATHS = ['static']               # folder `docs/static/` will be copied as‑is

############################
# Plugins (optional)       #
############################

# Pelican 4+ uses the “plugins” entry point.  If you need any plugins,
# install them via pip and list them here.
PLUGIN_PATHS = ['./plugins']
PLUGINS = ['ipynb.markup']
IPYNB_USE_METACELL = True

############################
# Miscellaneous settings   #
############################

# Feed generation is usually disabled for docs builds.
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None

# Relative URLs are handy for local builds (e.g. on Read the Docs).
RELATIVE_URLS = True

# Uncomment the following line to enable pagination (useful for blogs).
# DEFAULT_PAGINATION = 10

# If you use a custom Jinja2 filter or global, you can register it here.
# JINJA_FILTERS = {}

############################
# End of configuration     #
############################
