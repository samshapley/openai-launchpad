from wandb.sdk.data_types.trace_tree import Trace
import time

def wandb_span(span_kind, span_name, inputs={}, outputs={}, parent_span=None, root_span=None, status="success", metadata={}):
    end_time_ms = round(time.time() * 1000)

    # Ensure contiguous spans.
    if parent_span:
        start_time_ms = parent_span.end_time_ms  # start time is end time of parent span.
    elif root_span and not parent_span:
        start_time_ms = root_span.end_time_ms    # start time is end time of root span.
    else:
        start_time_ms = end_time_ms              # start time is now.

    span = Trace(
        kind=span_kind,
        name=span_name,
        inputs=inputs,
        outputs=outputs,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        status_code=status,
        metadata=metadata
    )

    # If parent span is provided, add the new span as a child and update end time.
    if parent_span:
        parent_span.add_child(span)          # New span becomes child of parent span.
        parent_span.end_time_ms = end_time_ms  # Always update end time if parent span is present.

    # If root span is provided
    if root_span: 
        if not parent_span:
            root_span.add_child(span)        # Only add new span as child of root if no parent span is provided.
            
        root_span.end_time_ms = end_time_ms  # Always update end time if root span is present.

    # If neither parent nor root span is provided, set root span to the new span.
    if not parent_span and not root_span:
        root_span = span  # If neither are provided, root span becomes the new span.

    return span, parent_span, root_span