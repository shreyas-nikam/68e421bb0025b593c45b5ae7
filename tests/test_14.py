import pytest
from definition_5780a4f7830e4abfaf196f45dd6f0a73 import create_interactive_widgets

@pytest.mark.parametrize("expected_keys", [
    (['sliders', 'dropdowns']),
    (['regularization_slider', 'sensitive_attribute_dropdown']),
    (['model_params', 'fairness_constraints']),
])

def test_create_interactive_widgets_structure(expected_keys):
    widgets = create_interactive_widgets()
    assert isinstance(widgets, dict)
    for key in expected_keys:
        assert key in widgets

def test_create_interactive_widgets_widget_types():
    widgets = create_interactive_widgets()
    for key, widget in widgets.items():
        assert hasattr(widget, 'value'), f"Widget {key} should have 'value' attribute"

def test_create_interactive_widgets_empty():
    widgets = create_interactive_widgets()
    assert len(widgets) > 0, "Widgets dictionary should not be empty"