"""
Run classification with Google Cloud.

Notes
-----
* Install Google Cloud SDK first: https://cloud.google.com/sdk/downloads?hl=de
* See https://cloud.google.com/translate/docs/detecting-language
"""

# Third party modules
import pkg_resources

# First party modules
import lidtk.classifiers


class GCClassifier(lidtk.classifiers.LIDClassifier):
    """LID with the Google Cloud classifier."""

    def predict(self, text: str) -> str:
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        # Third party modules
        from google.cloud import translate

        translate_client = translate.Client()
        result = translate_client.detect_language(text)
        return result["language"]


path = "classifiers/config/google-cloud.yaml"
filepath = pkg_resources.resource_filename("lidtk", path)
classifier = GCClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
entry_point = lidtk.classifiers.classifier_cli_factor(classifier)


@entry_point.command(name="list-languages")
def list_languages() -> None:
    """List all available languages."""
    # Third party modules
    from google.cloud import translate

    translate_client = translate.Client()

    results = translate_client.get_languages()

    for language in results:
        print("{name} ({language})".format(**language))
