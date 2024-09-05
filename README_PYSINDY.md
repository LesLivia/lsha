# Guide to Using PySINDy for System Identification

This guide explains how to use PySINDy for system identification in your case study.
## Step-by-step Instructions

1. **Configuring PySINDy in `config.ini`:**
   
To enable PySINDy, set the `ENABLE_PYSINDY` flag in the `config.ini` file:
    [DEFAULT] ENABLE_PYSINDY = True

When `ENABLE_PYSINDY` is set to `True`, the direct identification method will be applied in the `mi_query` process. This allows PySINDy to automatically identify and use the most relevant flow conditions for your case study.

2. **Using SUL Definition Instead of PySINDy:**

If you prefer to use the SUL (System Under Learning) definition of the case study rather than PySINDy's direct identification method, set `ENABLE_PYSINDY` to `False` in the `config.ini`:
    [DEFAULT] ENABLE_PYSINDY = False

In this case, you'll need to manually modify the flow conditions being used. The flow conditions identified by PySINDy are easy to spot, as they have different names compared to the standard ones. You should comment or uncomment these lines as needed to select the correct flow conditions for your analysis.

3. **Managing Flow Conditions:**

The flow conditions identified by PySINDy are specifically named and distinguished from others. You can find these flow conditions in the relevant configuration files. Depending on your needs:

- Comment out the standard flow conditions.
- Uncomment the PySINDy-generated flow conditions if you want to use them.

This process allows you to easily switch between using standard flow conditions or those identified by PySINDy, depending on the specific requirements of your analysis.

4. **Special Considerations for the MADE Case Study:**

In the case of the MADE study, two different methods are used for model extraction:

- **Single Model Extraction:** This method extracts a single model that encompasses the entire process. It is useful when you want a comprehensive model that covers all operations within the study.

- **Operation-Specific Model Extraction:** This method extracts a separate model for each known operation within the study. This approach is beneficial when you need detailed models for individual operations, allowing for more granular analysis.

Depending on your goals, you can choose to apply either the single model extraction or the operation-specific model extraction. Make sure to configure your settings accordingly in the `config.ini` and the relevant scripts.

By following these steps, you can efficiently manage how PySINDy is used within your case study, either by leveraging its direct identification capabilities, using predefined SUL definitions, or selecting the appropriate model extraction method for the MADE case study.


