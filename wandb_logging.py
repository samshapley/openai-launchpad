from wandb.sdk.data_types.trace_tree import Trace
import time

class SpanManager:
    def __init__(self, name="span_name"):
        self.span_name = name
        self.span_hierarchy = []

    def get_span_hierarchy(self):
        return self.span_hierarchy

    @staticmethod
    def generate_span_id(kind, name, end_time_ms):
        return f"{kind}-{name}-{end_time_ms}"
    
    @staticmethod
    def construct_span_tree_member(span_id, span, parent_span_id):
        return {"id": span_id, "span": span, "child_spans": [], "parent_span_id": parent_span_id}
    
    def _recursive_search(self, current_level, span_id):
        if isinstance(current_level, list):
            for span in current_level:
                if span["id"] == span_id:
                    return span
                else:
                    result = self._recursive_search(span["child_spans"], span_id)
                    if result:
                        return result
        return None
    
    def get_span_from_hierarchy(self, span_id):
        return self._recursive_search(self.span_hierarchy, span_id)
    
    def update_hierarchy_with_new_child(self, current_level, parent_span_id, new_child_span):
        """Recursively search the hierarchy for the parent span and add the new child span to its children."""
        if isinstance(current_level, list):
            for span in current_level:
                if span["id"] == parent_span_id:                   # Found the parent span
                    span["child_spans"].append(new_child_span)     # Add the new child span to hierarchy
                    span["span"].add_child(new_child_span["span"]) # Set the parent-child relationship in the span object
                    return True
                else:
                    if self.update_hierarchy_with_new_child(span["child_spans"], parent_span_id, new_child_span):
                        return True
        return False
        
    def update_ancestor_end_times(self, span_id, new_end_time_ms):
        span = self.get_span_from_hierarchy(span_id)  # Retrieve the span from the hierarchy
        if span:
            while span:
                span["span"].end_time_ms = new_end_time_ms  # Update the end time of the current span
                span = self.get_span_from_hierarchy(span["parent_span_id"])  # Move up to the parent span

    def add_span(self, span_id, span, parent_span_id=None):

        span_member = self.construct_span_tree_member(span_id, span, parent_span_id)

        if parent_span_id:
            # Update the hierarchy with the new child span
            if not self.update_hierarchy_with_new_child(self.span_hierarchy, parent_span_id, span_member):
                raise ValueError(f"Parent span with id {parent_span_id} not found.")
            # Update end times of direct ancestors
            self.update_ancestor_end_times(parent_span_id, span.end_time_ms)
        else:
            # Add new top-level span to the list
            if any(span["id"] == span_id for span in self.span_hierarchy):
                raise ValueError(f"Top-level span with id {span_id} already exists.")
            self.span_hierarchy.append(span_member)


    def wandb_span(self, span_kind, span_name, inputs={}, outputs={}, parent_span_id=None, status="success", metadata={}, span_id=None):
        end_time_ms = round(time.time() * 1000)

        # Generate span id if not provided.
        if not span_id:
            span_id = self.generate_span_id(span_kind, span_name, end_time_ms)

        # Add span_id to metadata.
        metadata["span_id"] = span_id

        # Ensure contiguous spans.
        if parent_span_id:
            parent_span = self.get_span_from_hierarchy(parent_span_id)["span"]
            start_time_ms = parent_span.end_time_ms  # start time is end time of parent span.

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

        # Add span to span hierarchy.
        self.add_span(span_id, span, parent_span_id) 

        return span_id
    
    def log_top_level_span(self):
        """
        Log only the top-level spans in the span hierarchy.
        """
        for span_member in self.span_hierarchy:
            span_member["span"].log(self.span_name)
    
    def log_all_spans(self, current_level=None):
        if current_level is None:
            current_level = self.span_hierarchy

        if isinstance(current_level, list):
            time.sleep(0.1)
            for span_member in current_level:
                span_member["span"].log(self.span_name)
                self.log_all_spans(span_member["child_spans"])