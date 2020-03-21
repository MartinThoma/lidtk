"""Utility functions for error analysis."""


def manual_error_analysis(errors, author_known_languages):
    """
    Print errors for manual analysis.

    Parameters
    ----------
    errors : dict of dict of lists
        Has the form 'errors[true][predicted] = [sample 1, sample 2, ...]'
    author_known_languages : list of str
    """
    # print("# Author knows the true language")
    # for true_lang in author_known_languages:
    #     if true_lang not in errors:
    #         continue
    #     for label, errors_lang in errors[true_lang].items():
    #         print("## True: {}, Predicted: {} (count: {})"
    #               .format(true_lang, label, len(errors_lang)))
    #         for nb, el in enumerate(errors_lang):
    #             print("{}. {}".format(nb, el))
    #     print("\n\n")

    print("# Author knows the (wrongly) predicted langauge")
    conv_err = {}
    for true_lang, tmp in errors.items():
        for pred_lang, samples in tmp.items():
            if pred_lang not in conv_err:
                conv_err[pred_lang] = {}
            if true_lang not in conv_err[pred_lang]:
                conv_err[pred_lang][true_lang] = []
            conv_err[pred_lang][true_lang] += samples

    for label in author_known_languages:
        if label not in errors:
            continue
        for true_lang, errors_lang in conv_err[label].items():
            print(
                "## True: {}, Predicted: {} (count: {})".format(
                    true_lang, label, len(errors_lang)
                )
            )
            for nb, el in enumerate(errors_lang):
                print("{}. {}".format(nb, el))
            print("")
        print("\n")
