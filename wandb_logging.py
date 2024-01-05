from wandb.sdk.data_types.trace_tree import Trace
import time
import copy

class WandbSpanManager:
    def __init__(self, name="span_name"):
        """
        Initializes the SpanManager with a default span name and an empty span hierarchy.

        Args:
            name (str): The name of the span. Defaults to 'span_name'.
        """
        self.span_name = name
        self.span_hierarchy = []

    def get_span_hierarchy(self):
        """
        Retrieves the current span hierarchy.

        Returns:
            list: The span hierarchy list.
        """
        return self.span_hierarchy

    @staticmethod
    def generate_span_id(kind, name, end_time_ms):
        """
        Generates a unique span ID based on the span kind, name, and end time.

        Args:
            kind (str): The kind of the span.
            name (str): The name of the span.
            end_time_ms (int): The end time of the span in milliseconds.

        Returns:
            str: A unique span ID.
        """
        return f"{kind}-{name}-{end_time_ms}"
    
    @staticmethod
    def construct_span_tree_member(span_id, span, parent_span_id):
        """
        Constructs a dictionary representing a member of the span tree.

        Args:
            span_id (str): The unique identifier for the span.
            span (Trace): The span object.
            parent_span_id (str): The unique identifier for the parent span.

        Returns:
            dict: A dictionary representing the span tree member.
        """
        return {"id": span_id, "span": span, "child_spans": [], "parent_span_id": parent_span_id}
    
    def _recursive_search(self, current_level, span_id):
        """
        Recursively searches for a span with the given ID in the current level of the span hierarchy.

        Args:
            current_level (list): The current level in the span hierarchy to search.
            span_id (str): The unique identifier for the span to find.

        Returns:
            dict or None: The found span dictionary or None if not found.
        """
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
        """
        Retrieves a span from the hierarchy by its ID.

        Args:
            span_id (str): The unique identifier for the span to retrieve.

        Returns:
            dict or None: The span dictionary if found, otherwise None.
        """
        return self._recursive_search(self.span_hierarchy, span_id)
    
    def update_hierarchy_with_new_child(self, current_level, parent_span_id, new_child_span):
        """
        Recursively searches the hierarchy for the parent span and adds the new child span to its children.

        Args:
            current_level (list): The current level in the span hierarchy to search.
            parent_span_id (str): The unique identifier for the parent span.
            new_child_span (dict): The new child span dictionary to add.

        Returns:
            bool: True if the child span was added successfully, False otherwise.
        """
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
        """
        Updates the end times of a span and all its ancestors in the hierarchy.

        Args:
            span_id (str): The unique identifier for the span to update.
            new_end_time_ms (int): The new end time in milliseconds to set for the span and its ancestors.
        """
        span = self.get_span_from_hierarchy(span_id)  # Retrieve the span from the hierarchy
        if span:
            while span:
                span["span"].end_time_ms = new_end_time_ms  # Update the end time of the current span
                span = self.get_span_from_hierarchy(span["parent_span_id"])  # Move up to the parent span

    def add_span(self, span_id, span, parent_span_id=None):
        """
        Adds a new span to the span hierarchy.

        This method constructs a span tree member and adds it to the hierarchy. If a parent span ID is provided,
        it updates the hierarchy with the new child span and updates the end times of direct ancestors. If no parent
        is provided, it adds a new top-level span to the hierarchy.

        Args:
            span_id (str): The unique identifier for the new span.
            span (Trace): The span object to add to the hierarchy.
            parent_span_id (str, optional): The unique identifier for the parent span. Defaults to None.

        Raises:
            ValueError: If the parent span ID is not found in the hierarchy or if a top-level span with the same ID already exists.
        """

        # Construct a new span tree member dictionary from the given span information.
        span_member = self.construct_span_tree_member(span_id, span, parent_span_id)

        if parent_span_id:
            # If a parent span ID is provided, attempt to add the new span as a child of the parent span.
            if not self.update_hierarchy_with_new_child(self.span_hierarchy, parent_span_id, span_member):
                # If the parent span cannot be found, raise an error.
                raise ValueError(f"Parent span with id {parent_span_id} not found.")
            # If the span was successfully added, update the end times of all ancestor spans.
            self.update_ancestor_end_times(parent_span_id, span.end_time_ms)
        else:
            # If no parent span ID is provided, ensure that a top-level span with the same ID does not already exist.
            if any(span["id"] == span_id for span in self.span_hierarchy):
                # If a top-level span with the same ID exists, raise an error.
                raise ValueError(f"Top-level span with id {span_id} already exists.")
            # Add the new span as a top-level span in the hierarchy.
            self.span_hierarchy.append(span_member)


    def wandb_span(self, span_kind, span_name, inputs={}, outputs={}, parent_span_id=None, status="success", metadata={}, span_id=None):
        """
        Creates and logs a new span with the provided details.

        This method generates a span ID if not provided, adds the span ID to the metadata, ensures contiguous spans by setting
        the start time to the end time of the parent span (if provided), and creates a new Trace object. It then adds the span
        to the span hierarchy and returns the span ID.

        Args:
            span_kind (str): The kind of the span.
            span_name (str): The name of the span.
            inputs (dict, optional): The inputs of the span. Defaults to an empty dictionary.
            outputs (dict, optional): The outputs of the span. Defaults to an empty dictionary.
            parent_span_id (str, optional): The unique identifier for the parent span. Defaults to None.
            status (str, optional): The status of the span. Defaults to "success".
            metadata (dict, optional): Additional metadata for the span. Defaults to an empty dictionary.
            span_id (str, optional): The unique identifier for the span. If not provided, it will be generated. Defaults to None.

        Returns:
            str: The unique identifier for the created span.
        """

        # Get the current time in milliseconds to be used as the end time for the span.
        end_time_ms = round(time.time() * 1000)

        # If a span ID is not provided, generate one using the span kind, name, and end time.
        if not span_id:
            span_id = self.generate_span_id(span_kind, span_name, end_time_ms)

        # Create a deep copy of the metadata and add the span ID to it.
        metadata_with_id = copy.deepcopy(metadata)
        metadata_with_id["span_id"] = span_id

        # If a parent span ID is provided, set the start time of the new span to the end time of the parent span.
        if parent_span_id:
            parent_span = self.get_span_from_hierarchy(parent_span_id)["span"]
            start_time_ms = parent_span.end_time_ms
        else:
            # If no parent span ID is provided, set the start time of the new span to the current time.
            start_time_ms = end_time_ms

        # Create a new Trace object with the provided details and the calculated start and end times.
        span = Trace(
            kind=span_kind,
            name=span_name,
            inputs=inputs,
            outputs=outputs,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            status_code=status,
            metadata=metadata_with_id
        )

        # Add the new span to the span hierarchy.
        self.add_span(span_id, span, parent_span_id) 

        # Return the unique identifier of the created span.
        return span_id
    
    def log_top_level_span(self):
        """
        Logs only the top-level spans in the span hierarchy.

        This method iterates over the top-level spans in the span hierarchy and logs each span using its `log` method.
        """
        for span_member in self.span_hierarchy:  # Iterate through each top-level span in the hierarchy
            span_member["span"].log(self.span_name)  # Call the log method on the Trace object of each top-level span

    def log_all_spans(self, current_level=None):
        """
        Recursively logs all spans in the span hierarchy.

        This method traverses the span hierarchy and logs each span. It recursively logs child spans for each span in the hierarchy.

        Args:
            current_level (list, optional): The current level in the span hierarchy to log. If None, it starts with the top-level spans. Defaults to None.
        """
        if current_level is None:
            current_level = self.span_hierarchy  # If no specific level is provided, start with the top-level spans
        if isinstance(current_level, list):  # Ensure the current level is a list before iterating
            for span_member in current_level:  # Iterate through each span in the current level
                span_member["span"].log(self.span_name)  # Log the current span
                self.log_all_spans(span_member["child_spans"])  # Recursively log all child spans

    def _replace_span_in_hierarchy(self, span_id, updated_span):
        """
        Replaces a span in the hierarchy with an updated span.

        Args:
            span_id (str): The unique identifier for the span to replace.
            updated_span (Trace): The updated span object.
        """
        def _recursive_replace(current_level, span_id, updated_span):
            for i, span_member in enumerate(current_level):
                if span_member["id"] == span_id:
                    current_level[i]["span"] = updated_span
                    return True
                else:
                    if _recursive_replace(span_member["child_spans"], span_id, updated_span):
                        return True
            return False

        if not _recursive_replace(self.span_hierarchy, span_id, updated_span):
            raise ValueError(f"Span with id {span_id} not found in the hierarchy.")
        
    def update_span_by_id(self, span_id, inputs=None, outputs=None, metadata=None):
        """
        Updates the inputs, outputs, and metadata of a span with the given span_id in the span hierarchy.
        Doesn't overwrite existing inputs, outputs, or metadata.

        Args:
            span_id (str): The unique identifier for the span to update.
            inputs (dict, optional): The new inputs to add for the span.
            outputs (dict, optional): The new outputs to add for the span.
            metadata (dict, optional): The new metadata to add for the span.
        """
        # Retrieve the span from the hierarchy using its ID.
        span_member = self.get_span_from_hierarchy(span_id)
        if span_member:
            # Update the inputs and outputs of the span if provided.
            if inputs is not None or outputs is not None:
                span_member["span"].add_inputs_and_outputs(inputs or {}, outputs or {})
            # Update the metadata of the span if provided.
            if metadata is not None:
                span_member["span"].add_metadata(metadata)
            # Replace the span in the hierarchy with the updated span.
            self._replace_span_in_hierarchy(span_id, span_member["span"])
        else:
            # If the span is not found, raise an error.
            raise ValueError(f"Span with id {span_id} not found.")