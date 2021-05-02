# Python Built-Ins:
from collections import namedtuple
import traceback

# External Dependencies:
from IPython.display import Code, display, HTML
import ipywidgets as widgets

# RGB 0-255 color for highlighting (pick something that works with both dark and light!)
HIGHLIGHT_COLOR = [64, 164, 64];

# Dummy update object for forcing initial widget loads:
DummyUpdate = namedtuple("DummyUpdate", ["new"])

# TODO: Type annotations

# TODO: Span opacity by answer confidence

# TODO: Accept multiple answers and highlight most confident differently

# TODO: If QAs are in the dataset, pre-populate the textbox with one and also hilite the ground truth answer

def is_sequence(arg):
    """Sufficiently list-like and not a string or an object"""
    return (
        (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
        and not
        (hasattr(arg, "strip") or hasattr(arg, "keys"))
    )


def get_highlight_color(confidence=1, rgb_uint8=HIGHLIGHT_COLOR):
    """Generate a CSS color spec for highlighting an answer with given `confidence`

    `confidence` sets highlight opacity, but with a bit of a boost so that even very non-confident
    predictions are still visible.
    """
    alpha_lim = 0.2
    alpha_frac = alpha_lim + min(1., max(0., confidence)) * (1. - alpha_lim)
    return f"rgba({rgb_uint8[0]}, {rgb_uint8[1]}, {rgb_uint8[2]}, {alpha_frac})"


def dummy_answer_fetcher(context, question):
    """A dummy answer fetcher function used to illustrate the interface

    The raw_response return value is optional: Can return just a start/end tuple instead.

    This dummy function simply sleeps for a second and then returns a random range with a dummy raw response.

    Parameters
    ----------
    context : str
        The text to extract the answer from (i.e. paragraph of a document)
    question : str
        The text of the query

    Returns
    -------
    location : Tuple[int, int]
        Starting and ending index (by character) of the answer extracted from the text
    raw_response : str
        Raw response to display in the debug output widget
    """
    import json
    import time
    import random
    time.sleep(1.)
    ntotal = len(context)
    a, b = random.random(), random.random()
    ixstart = max(0, min(ntotal - 5, int(a * ntotal)))
    ixend = max(ixstart + 1, min(ntotal - 1, ixstart + int(b * (ntotal - ixstart))))
    dummyresult = {
        "message": "I am a dummy API response!",
        "query": question,
        "context": context,
    }
    return (ixstart, ixend), json.dumps(dummyresult)


def squad_widget(data, answer_fetcher):
    """Generate an interactive widget to query SQuAD `data` with `answer_fetcher`

    Parameters
    ----------
    data : Union[object, List[object]]
        Loaded SQuAD JSON - either in its entirety or just the "data" array.
    answer_fetcher : Callable[[str, str], Union[Tuple[int, int], Tuple[Tuple[int, int], str]]]
        Function to find the span of the selected context that answers the submitted question; returning
        character-wise start and end indexes, and optionally also the raw response body to display.

    Returns
    -------
    result : ipywidgets.interact
        An interactive widget to be rendered in Jupyter/JupyterLab
    """
    # Allow passing either an entire SQuAD JSON or a list of SQuAD examples:
    if not is_sequence(data) and data.get("data") is not None:
        data = data["data"]

    assert len(data), "List of documents `data` cannot be empty!"

    # Set up our widgets:
    docslider = widgets.IntSlider(description="Document", value=0, min=0, max=len(data) - 1)
    doctitle = widgets.Output()
    paraslider = widgets.IntSlider(
        description="Paragraph",
        value=0,
        min=0,
        max=len(data[0]["paragraphs"]),
    )
    question = widgets.Text(
        description="Question",
        placeholder="Type a question...",
        layout=widgets.Layout(width="90%"),
    )
    askbutton = widgets.Button(description="Ask!")
    paracontent = widgets.Output()
    output = widgets.Output(layout=widgets.Layout(border="1px solid #999"))
    # We use display(Code()) because output.append_stdout contents don't seem to clear properly:
    # https://github.com/jupyter-widgets/ipywidgets/issues/2584
    with output:
        display(Code("Ready", language="text"))

    def update_para(change):
        """Listen and respond to paragraph selector changes"""
        context = data[docslider.value]["paragraphs"][change.new]["context"]
        paracontent.clear_output(wait=True)
        with paracontent:
            display(HTML(f"<p>{context}</p>"))

    paraslider.observe(update_para, "value")
    update_para(DummyUpdate(0))

    def update_doc(change):
        """Listen and respond to document selector changes"""
        doc = data[change.new]
        doctitle.clear_output(wait=True)
        with doctitle:
            display(HTML(f'<strong>{doc["title"]}</strong>'))
        # Preserve the value of the paragraph slider where possible:
        n_paras = len(doc["paragraphs"])
        if (paraslider.value >= n_paras):
            paraslider.value = n_paras - 1
        else:
            update_para(DummyUpdate(paraslider.value))
        paraslider.max = n_paras - 1

    docslider.observe(update_doc, "value")
    update_doc(DummyUpdate(0))

    def ask(_):
        """Handle question requests"""
        output.clear_output(wait=True)
        with output:
            display(Code("Asking...", language="text"))
        context = data[docslider.value]["paragraphs"][paraslider.value]["context"]

        try:
            results_raw = answer_fetcher(context, question.value)
        except Exception as err:
            output.clear_output(wait=True)
            output.append_stderr("Failed to call answer fetcher\n")
            output.append_stderr(traceback.format_exc())
            return

        output.clear_output(wait=True)
        if not is_sequence(results_raw):
            # TODO: Proper way of displaying errors in callbacks
            output.append_stderr(
                "ValueError: answer_fetcher fn must return a tuple startix, endix; and optionally also a "
                f"raw response. Got {results_raw}\n"
            )
        if is_sequence(results_raw[0]):
            # First result is a (startix, endix) tuple, second result is the raw output
            ixstart, ixend = results_raw[0]
            rawres = results_raw[1] if len(results_raw) > 1 else None
        else:
            # Just a tuple (startix, endix)... May be extended with the raw output as third element
            ixstart = results_raw[0]
            ixend = results_raw[1]
            rawres = results_raw[2] if len(results_raw) > 2 else None

        if ixstart < 0:
            output.append_stderr(f"Warn: Negative ixstart {ixstart} will be overridden to 0\n")
            ixstart = 0
        if ixend >= len(context):
            output.append_stderr(
                f"Warn: ixend {ixend} greater than context length {len(context)} and will be overridden.\n"
            )

        if rawres is not None:
            with output:
                display(Code(f"Raw result:\n{rawres}\n", language="text"))
            if isinstance(rawres, dict):
                confidence = rawres.get("confidence", rawres.get("score"))
            else:
                confidence = None
        else:
            confidence = None
        prestart = context[:ixstart]
        answer = context[ixstart:ixend]
        postend = context[ixend:]
        paracontent.clear_output(wait=True)
        with paracontent:
            display(HTML(
                "".join((
                    "<p>",
                    prestart,
                    f'<span style="background: {get_highlight_color(confidence=confidence)};">',
                    answer,
                    "</span>",
                    postend,
                    "</p>"
                ))
            ))

    askbutton.on_click(ask)

    return widgets.VBox([
        widgets.HTML(
            '<p><b>🔮 SQuAD Explorer: 🔍</b> Select a document and paragraph; type a question and click Ask!</p>'
        ),
        widgets.HBox([docslider, doctitle]),
        paraslider,
        widgets.HBox([question, askbutton]),
        paracontent,
        output
    ])


def qna_widget(answer_fetcher, default_context=""):
    """Generate an interactive widget to query a context paragraph with `answer_fetcher`

    Parameters
    ----------
    answer_fetcher : Callable[[str, str], Union[Tuple[int, int], Tuple[Tuple[int, int], str]]]
        Function to find the span of the selected context that answers the submitted question; returning
        character-wise start and end indexes, and optionally also the raw response body to display.
    default_context : str
        Optional 'context' (source) paragraph to pre-populate for question answering

    Returns
    -------
    result : ipywidgets.interact
        An interactive widget to be rendered in Jupyter/JupyterLab
    """
    # Set up our widgets:
    question = widgets.Text(
        description="Question",
        placeholder="Type a question...",
        layout=widgets.Layout(width="90%"),
    )
    askbutton = widgets.Button(description="Ask!")
    parainput = widgets.Textarea(
        value=default_context,
        placeholder="Enter the paragraph containing the answer to your question",
        description="Context:",
        layout=widgets.Layout(width="99%"),
    )
    paracontent = widgets.Output()
    output = widgets.Output(layout=widgets.Layout(border="1px solid #999"))
    # We use display(Code()) because output.append_stdout contents don't seem to clear properly:
    # https://github.com/jupyter-widgets/ipywidgets/issues/2584
    with output:
        display(Code("Ready", language="text"))

    with paracontent:
        display(HTML(f"<p>{default_context}</p>"))

    def ask(_):
        """Handle question requests"""
        output.clear_output(wait=True)
        with output:
            display(Code("Asking...", language="text"))

        context = parainput.value
        with paracontent:
            display(HTML(f"<p>{context}</p>"))

        try:
            results_raw = answer_fetcher(context, question.value)
        except Exception as err:
            output.clear_output(wait=True)
            output.append_stderr("Failed to call answer fetcher\n")
            output.append_stderr(traceback.format_exc())
            return

        output.clear_output(wait=True)
        if not is_sequence(results_raw):
            # TODO: Proper way of displaying errors in callbacks
            output.append_stderr(
                "ValueError: answer_fetcher fn must return a tuple startix, endix; and optionally also a "
                f"raw response. Got {results_raw}\n"
            )
        if is_sequence(results_raw[0]):
            # First result is a (startix, endix) tuple, second result is the raw output
            ixstart, ixend = results_raw[0]
            rawres = results_raw[1] if len(results_raw) > 1 else None
        else:
            # Just a tuple (startix, endix)... May be extended with the raw output as third element
            ixstart = results_raw[0]
            ixend = results_raw[1]
            rawres = results_raw[2] if len(results_raw) > 2 else None

        if ixstart < 0:
            output.append_stderr(f"Warn: Negative ixstart {ixstart} will be overridden to 0\n")
            ixstart = 0
        if ixend >= len(context):
            output.append_stderr(
                f"Warn: ixend {ixend} greater than context length {len(context)} and will be overridden.\n"
            )

        if rawres is not None:
            with output:
                display(Code(f"Raw result:\n{rawres}\n", language="text"))
            if isinstance(rawres, dict):
                confidence = rawres.get("confidence", rawres.get("score"))
            else:
                confidence = None
        else:
            confidence = None
        prestart = context[:ixstart]
        answer = context[ixstart:ixend]
        postend = context[ixend:]
        paracontent.clear_output(wait=True)
        with paracontent:
            display(HTML(
                "".join((
                    "<p>",
                    prestart,
                    f'<span style="background: {get_highlight_color(confidence=confidence)};">',
                    answer,
                    "</span>",
                    postend,
                    "</p>"
                ))
            ))

    askbutton.on_click(ask)

    return widgets.VBox([
        widgets.HTML(
            '<p><b>🔮 Q&A Explorer: 🔍</b> Enter some source text; type a question and click Ask!</p>'
        ),
        parainput,
        widgets.HBox([question, askbutton]),
        paracontent,
        output
    ])
