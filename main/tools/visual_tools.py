import http.server
import socketserver
from tools.config_tools import PATHS
from tools.file_system_tools import FS
from flask import Flask, jsonify
import logging
import os

# disable excessive logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
os.environ['WERKZEUG_RUN_MAIN'] = 'true'

# 
# data setup
# 
_json_data = {}
_app = Flask(__name__)
_display_format = {}
_javascript_code = lambda : """
    <style>
        /* This is just basline CSS from: https://www.npmjs.com/package/css-baseline */
        @-ms-viewport {
            width: device-width;
        }
        article,
        aside,
        details,
        figcaption,
        figure,
        footer,
        header,
        hgroup,
        menu,
        nav,
        section,
        main,
        summary {
            display: block;
        }

        *,
        *::before,
        *::after {
            box-sizing: inherit;
        }

        html {
            /* 1 */
            box-sizing: border-box;
            /* 2 */
            touch-action: manipulation;
            /* 3 */
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
            /* 4 */
            -ms-overflow-style: scrollbar;
            /* 5 */
            -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
        }

        body {
            line-height: 1;
        }

        html,
        body,
        div,
        span,
        applet,
        object,
        iframe,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        p,
        blockquote,
        pre,
        a,
        abbr,
        acronym,
        address,
        big,
        cite,
        code,
        del,
        dfn,
        em,
        img,
        ins,
        kbd,
        q,
        s,
        samp,
        small,
        strike,
        strong,
        sub,
        sup,
        tt,
        var,
        b,
        u,
        i,
        center,
        dl,
        dt,
        dd,
        ol,
        ul,
        li,
        fieldset,
        form,
        label,
        legend,
        table,
        caption,
        tbody,
        tfoot,
        thead,
        tr,
        th,
        td,
        article,
        aside,
        canvas,
        details,
        embed,
        figure,
        figcaption,
        footer,
        header,
        hgroup,
        menu,
        nav,
        output,
        ruby,
        section,
        summary,
        time,
        mark,
        audio,
        video,
        main {
            font-size: 100%;
            font: inherit;
            vertical-align: baseline;
        }

        ol,
        ul {
            list-style: none;
        }

        blockquote,
        q {
            quotes: none;
        }

        blockquote::before,
        blockquote::after,
        q::before,
        q::after {
            content: "";
            content: none;
        }

        table {
            border-collapse: collapse;
            border-spacing: 0;
        }

        hr {
            /* 1 */
            box-sizing: content-box;
            height: 0;
            /* 2 */
            overflow: visible;
        }

        pre,
        code,
        kbd,
        samp {
            /* 1 */
            font-family: monospace, monospace;
        }

        pre {
            /* 2 */
            overflow: auto;
            /* 3 */
            -ms-overflow-style: scrollbar;
        }

        a {
            /* 1 */
            background-color: transparent;
            /* 2 */
            -webkit-text-decoration-skip: objects;
        }

        abbr[title] {
            /* 1 */
            border-bottom: none;
            /* 2 */
            text-decoration: underline;
            text-decoration: underline dotted;
        }

        b,
        strong {
            font-weight: bolder;
        }

        small {
            font-size: 80%;
        }

        sub,
        sup {
            font-size: 75%;
            line-height: 0;
            position: relative;
        }

        sub {
            bottom: -0.25em;
        }

        sup {
            top: -0.5em;
        }

        img {
            border-style: none;
        }

        svg:not(:root) {
            overflow: hidden;
        }

        button {
            border-radius: 0;
        }

        input,
        button,
        select,
        optgroup,
        textarea {
            font-family: inherit;
            font-size: inherit;
            line-height: inherit;
        }

        button,
        [type="reset"],
        [type="submit"],
        html [type="button"] {
            -webkit-appearance: button;
        }

        input[type="date"],
        input[type="time"],
        input[type="datetime-local"],
        input[type="month"] {
            -webkit-appearance: listbox;
        }

        fieldset {
            min-width: 0;
        }

        [tabindex="-1"]:focus {
            outline: 0 !important;
        }

        button,
        input {
            overflow: visible;
        }

        button,
        select {
            text-transform: none;
        }

        button::-moz-focus-inner,
        [type="button"]::-moz-focus-inner,
        [type="reset"]::-moz-focus-inner,
        [type="submit"]::-moz-focus-inner {
            border-style: none;
            padding: 0;
        }

        legend {
            /* 1 */
            max-width: 100%;
            white-space: normal;
            /* 2 */
            color: inherit;
            /* 3 */
            display: block;
        }

        progress {
            vertical-align: baseline;
        }

        textarea {
            overflow: auto;
        }

        [type="checkbox"],
        [type="radio"] {
            /* 1 */
            box-sizing: border-box;
            /* 2 */
            padding: 0;
        }

        [type="number"]::-webkit-inner-spin-button,
        [type="number"]::-webkit-outer-spin-button {
            height: auto;
        }

        [type="search"] {
            /* 1 */
            -webkit-appearance: textfield;
            /* 2 */
            outline-offset: -2px;
        }

        [type="search"]::-webkit-search-cancel-button,
        [type="search"]::-webkit-search-decoration {
            -webkit-appearance: none;
        }

        ::-webkit-file-upload-button {
            /* 1 */
            -webkit-appearance: button;
            /* 2 */
            font: inherit;
        }

        template {
            display: none;
        }

        [hidden] {
            display: none;
        }

        button:focus {
            outline: 1px dotted;
            outline: 5px auto -webkit-focus-ring-color;
        }

        button:-moz-focusring,
        [type="button"]:-moz-focusring,
        [type="reset"]:-moz-focusring,
        [type="submit"]:-moz-focusring {
            outline: 1px dotted ButtonText;
        }

        html,
        body,
        div,
        span,
        applet,
        object,
        iframe,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        p,
        blockquote,
        pre,
        a,
        abbr,
        acronym,
        address,
        big,
        cite,
        code,
        del,
        dfn,
        em,
        img,
        ins,
        kbd,
        q,
        s,
        samp,
        small,
        strike,
        strong,
        sub,
        sup,
        tt,
        var,
        b,
        u,
        i,
        center,
        dl,
        dt,
        dd,
        ol,
        ul,
        li,
        fieldset,
        form,
        label,
        legend,
        table,
        caption,
        tbody,
        tfoot,
        thead,
        tr,
        th,
        td,
        article,
        aside,
        canvas,
        details,
        embed,
        figure,
        figcaption,
        footer,
        header,
        hgroup,
        menu,
        nav,
        output,
        ruby,
        section,
        summary,
        time,
        mark,
        audio,
        video,
        main {
            margin: 0;
            padding: 0;
            border: 0;
        }

        input,
        button,
        select,
        optgroup,
        textarea {
            margin: 0;
        }

        body {
            width: 100vw;
            min-height: 100vh;
            overflow: visible;
            scroll-behavior: auto;
        }

        textarea {
            resize: vertical;
        }

        br {
            display: block;
            content: "";
            border-bottom: 0px solid transparent;
        }

        h1 {
            font-size: 3.78rem;
        }

        h2 {
            font-size: 3.204rem;
        }

        h3 {
            font-size: 2.628rem;
        }

        h4 {
            font-size: 2.052rem;
        }

        h5 {
            font-size: 1.476rem;
        }

        h6 {
            font-size: 1.15rem;
        }

        body {
            font-family: sans-serif;
        }

    </style>
    """+f"""
    <script>
        {FS.read(PATHS["visualizer"])}
    </script>
"""

# 
# setup data endpoints
# 
@_app.route("/")
def ___entrypoint___():
    # FIXME: make this constant again
    return _javascript_code()

@_app.get("/data")
def ___data___():
    global _json_data
    return jsonify(_json_data)

@_app.route("/method")
def ___method___():
    global _display_method
    return jsonify(_display_method)

# 
# internal method for getting data
# 
def display(data_to_display, display_method):
    global _json_data
    global _display_format
    # 
    # arg setup
    # 
    if True:
        if display_method is None:
            display_method = {}
        print('display_method = ', display_method)
        # defaults 
        display_method = {
            "port": 8900,
            "host": '0.0.0.0',
            **display_method,
        }
    print('display_method = ', display_method)
    # 
    # transfer data
    # 
    _display_format = display_method
    _json_data = data_to_display
    # 
    # start server
    # 
    _app.run(host=_display_format["host"], port=_display_format["port"])

# display({
#     "x": [1,2,3,4,],
#     "y": [1,2,3,4,],
# }, display_method={
#     "plot": "scatter",
# })