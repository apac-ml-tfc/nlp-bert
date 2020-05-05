# Python Built-Ins:
from collections import namedtuple

# External Dependencies:
from IPython.display import display, HTML
import ipywidgets as widgets

# Dummy update object for forcing initial widget loads:
DummyUpdate = namedtuple("DummyUpdate", ["new"])

# TODO: Type annotations

# TODO: Span opacity by answer confidence

# TODO: Accept multiple answers and highlight most confident differently

# TODO: If QAs are in the dataset, pre-populate the textbox with one and also hilite the ground truth answer

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
    if data.get("data") is not None:
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

    def update_para(change):
        """Listen and respond to paragraph selector changes"""
        context = data[docslider.value]["paragraphs"][change.new]["context"]
        paracontent.clear_output()
        with paracontent:
            display(HTML(f"<p>{context}</p>"))

    paraslider.observe(update_para, "value")
    update_para(DummyUpdate(0))

    def update_doc(change):
        """Listen and respond to document selector changes"""
        doc = data[change.new]
        doctitle.clear_output()
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
        output.clear_output()
        context = data[docslider.value]["paragraphs"][paraslider.value]["context"]

        try:
            results_raw = answer_fetcher(context, question.value)
        except:
            # TODO: Better stack trace
            output.append_stderr("Failed to call answer fetcher")
            return

        if not hasattr(results_raw, "__getitem__"):
            # TODO: Proper way of displaying errors in callbacks
            output.append_stderr(
                "ValueError: answer_fetcher fn must return a tuple startix, endix; and optionally also a "
                f"raw response. Got {results_raw}\n"
            )
        if hasattr(results_raw[0], "__getitem__"):
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
            output.append_stdout(f"Raw result:\n{rawres}")
        prestart = context[:ixstart]
        answer = context[ixstart:ixend]
        postend = context[ixend:]
        paracontent.clear_output()
        with paracontent:
            display(HTML(
                "".join((
                    "<p>",
                    prestart,
                    '<span style="background: lime;">',
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
