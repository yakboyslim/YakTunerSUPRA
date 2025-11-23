import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import re
import traceback
import sys
import difflib
import requests
from st_copy_button import st_copy_button
from io import BytesIO
from scipy import interpolate

# --- Add project root to sys.path ---
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Import the custom tuning modules ---
from WG import run_wg_analysis
from MFF import run_mff_analysis
from KNK import run_knk_analysis
from tuning_loader import TuningData
from error_reporter import send_to_google_sheets

# --- Constants ---
default_vars = "variables.csv"
XDF_MAP_LIST_CSV = 'maps_to_parse.csv'
LOG_METADATA_ROWS_TO_SKIP = 4
GITHUB_TREES_API = "https://api.github.com/repos/dmacpro91/BMW-XDFs/git/trees/master?recursive=1"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dmacpro91/BMW-XDFs/master"

# --- Page Configuration ---
st.set_page_config(
    page_title="YAKtuner Online",
    layout="wide"
)

st.title("☁️ YAKtuner Online")


# --- Helper & Core Logic Functions ---

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_firmwares_from_github():
    """
    Recursively scans the entire GitHub repo for .xdf files and builds a
    map of firmware IDs to their full paths.
    """
    headers = {}
    # Try to get the token from Streamlit's secrets manager
    if 'GITHUB_TOKEN' in st.secrets:
        headers['Authorization'] = f"token {st.secrets['GITHUB_TOKEN']}"
        print("Using GitHub token for API request.")
    else:
        print("Warning: GITHUB_TOKEN not found in secrets. Making unauthenticated request.")

    try:
        response = requests.get(GITHUB_TREES_API, headers=headers)
        response.raise_for_status()
        tree = response.json().get('tree', [])

        firmware_map = {}
        for item in tree:
            path = item.get('path')
            if path and path.endswith('.xdf'):
                # Firmware ID is the filename without extension
                firmware_id = os.path.splitext(os.path.basename(path))[0]
                firmware_map[firmware_id] = path

        if not firmware_map:
            st.warning("No .xdf files found in the GitHub repository.")
            return {}

        return firmware_map
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch firmware list from GitHub: {e}")
        return {}


@st.cache_data(ttl=86400)  # Cache for 1 day
def download_xdf(_xdf_path):
    """
    Downloads the XDF file from the given full path in the repo.
    Caches the result to avoid re-downloading.
    """
    if not _xdf_path:
        return None
    url = f"{GITHUB_RAW_BASE}/{_xdf_path}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download XDF from path {_xdf_path}: {e}")
        return None


def get_firmware_from_log(log_file):
    """
    Parses the uploaded log file to find the Ecu PRGID.
    """
    try:
        # Read the first few lines to find the metadata
        content = log_file.read(4096).decode('latin1')  # Read first 4KB
        log_file.seek(0)  # IMPORTANT: Reset pointer for later use
        match = re.search(r'#Ecu PRGID:\s*([A-Fa-f0-9]+)', content)
        if match:
            return match.group(1)
        return None
    except Exception as e:
        print(f"Error parsing log for firmware: {e}")
        return None


def display_table_with_copy_button(title: str, styled_df, raw_df: pd.DataFrame):
    """
    Displays a title, a styled DataFrame, and a button to copy raw data.
    """
    st.write(title)
    clipboard_text = raw_df.to_csv(sep='\t', index=False, header=False)
    st.dataframe(styled_df)
    button_label = f"📋 Copy {title.strip('# ')} Data"
    button_key = f"copy_btn_{re.sub(r'[^a-zA-Z0-9]', '', title)}"
    st_copy_button(clipboard_text, button_label, key=button_key)
    st.caption("Use the button above to copy data for pasting into TunerPro.")


def normalize_header(header_name):
    """Normalizes a log file header for case-insensitive and unit-agnostic comparison."""
    normalized = re.sub(r'\s*\([^)]*\)\s*$', '', str(header_name))
    return normalized.lower().strip()


def _find_alias_match(aliases, log_headers):
    """Finds an exact match for a list of aliases within the log headers."""
    normalized_aliases = {normalize_header(a) for a in aliases}
    for header in log_headers:
        if normalize_header(header) in normalized_aliases:
            return header
    return None


def _find_best_match(target_name, log_headers, cutoff=0.7):
    """Finds the best fuzzy match for a target name from a list of log headers."""
    normalized_target = normalize_header(target_name)
    normalized_headers = {h: normalize_header(h) for h in log_headers}
    search_space = list(normalized_headers.values())
    best_matches = difflib.get_close_matches(normalized_target, search_space, n=1, cutoff=cutoff)
    if best_matches:
        for original_header, normalized_header in normalized_headers.items():
            if normalized_header == best_matches[0]:
                return original_header
    return None


def map_log_variables_streamlit(log_df, varconv_df):
    """
    Performs a 3-tiered, robust, automatic, and interactive variable mapping.
    """
    if 'mapping_initialized' not in st.session_state:
        with st.status("Automatically mapping log variables...", expanded=True) as status:
            st.session_state.mapping_initialized = True
            st.session_state.mapping_complete = False
            st.session_state.vars_to_map = []
            st.session_state.varconv_array = varconv_df.to_numpy()
            st.session_state.log_df_mapped = log_df.copy()

            varconv = st.session_state.varconv_array
            available_log_headers = log_df.columns.tolist()
            missing_vars_indices = []
            found_vars_indices = set()

            # Pass 1: Prioritize Exact Alias Matches
            for i in range(1, varconv.shape[1]):
                aliases_str = str(varconv[0, i])
                canonical_name = varconv[1, i]
                aliases = aliases_str.split(',')
                alias_match = _find_alias_match(aliases, available_log_headers)

                if alias_match:
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={alias_match: canonical_name}
                    )
                    st.session_state.varconv_array[0, i] = alias_match
                    available_log_headers.remove(alias_match)
                    found_vars_indices.add(i)

            # Pass 2: Fuzzy Matching for Remaining Variables
            for i in range(1, varconv.shape[1]):
                if i in found_vars_indices:
                    continue

                canonical_name = varconv[1, i]
                friendly_name = varconv[2, i] if varconv.shape[0] > 2 and pd.notna(varconv[2, i]) else canonical_name
                fuzzy_match = _find_best_match(friendly_name, available_log_headers)

                if fuzzy_match:
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={fuzzy_match: canonical_name}
                    )
                    st.session_state.varconv_array[0, i] = fuzzy_match
                    available_log_headers.remove(fuzzy_match)
                    found_vars_indices.add(i)
                else:
                    missing_vars_indices.append(i)

            st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

            if not missing_vars_indices:
                st.session_state.mapping_complete = True
                status.update(label="Variable mapping complete.", state="complete", expanded=False)
            else:
                st.session_state.vars_to_map = missing_vars_indices
                status.update(label="Manual input required...", state="complete", expanded=True)

    if st.session_state.get('vars_to_map'):
        varconv = st.session_state.varconv_array
        current_var_index = st.session_state.vars_to_map[0]
        prompt_name = varconv[2, current_var_index] if varconv.shape[0] > 2 and pd.notna(
            varconv[2, current_var_index]) else varconv[1, current_var_index]

        st.warning(f"Could not find a match for: **{prompt_name}**")
        st.write(
            f"Please select the corresponding column from your log file. The last known name was `{varconv[0, current_var_index]}`.")

        with st.form(key=f"mapping_form_{current_var_index}"):
            options = ['[Not Logged]'] + log_df.columns.tolist()
            selected_header = st.selectbox("Select Log File Column:", options=options)
            submitted = st.form_submit_button("Confirm Mapping")

            if submitted:
                if selected_header != '[Not Logged]':
                    st.session_state.log_df_mapped = st.session_state.log_df_mapped.rename(
                        columns={selected_header: varconv[1, current_var_index]}
                    )
                    st.session_state.varconv_array[0, current_var_index] = selected_header

                st.session_state.vars_to_map.pop(0)
                if not st.session_state.vars_to_map:
                    st.session_state.mapping_complete = True
                    st.session_state.updated_varconv_df = pd.DataFrame(st.session_state.varconv_array)

                st.rerun()
        return None

    if st.session_state.get('mapping_complete'):
        return st.session_state.log_df_mapped

    return None


# This function is NOT cached to ensure it always runs with the latest XDF content.
def load_all_maps_streamlit(bin_content, xdf_content, xdf_name):
    """Loads all ECU maps from file contents. This function is NOT cached."""
    st.write("Loading tune data from binary file...")
    try:
        loader = TuningData(bin_content)
    except Exception as e:
        st.error(f"Failed to initialize the binary file loader. Error: {e}")
        return None

    if xdf_content is None:
        st.error(f"Cannot parse maps because XDF content is missing for {xdf_name}.")
        return None

    st.write(f"Parsing maps from XDF: {xdf_name}...")
    tmp_xdf_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xdf") as tmp_xdf:
            tmp_xdf.write(xdf_content)
            tmp_xdf_path = tmp_xdf.name
        loader.load_from_xdf(tmp_xdf_path, XDF_MAP_LIST_CSV)
    except Exception as e:
        st.error(f"Failed to parse XDF file.")
        st.exception(e)
        return None
    finally:
        if os.path.exists(tmp_xdf_path):
            os.remove(tmp_xdf_path)

    all_maps = loader.maps
    if not all_maps:
        st.error("No maps were loaded. Check your XDF and the 'maps_to_parse.csv' file.")
        return None

    st.write(f"Successfully loaded {len(all_maps)} maps into memory.")
    with st.expander("Click to view list of all loaded maps"):
        sorted_map_names = sorted(list(all_maps.keys()))
        num_columns = 3
        columns = st.columns(num_columns)
        maps_per_column = (len(sorted_map_names) + num_columns - 1) // num_columns
        for i in range(num_columns):
            with columns[i]:
                for map_name in sorted_map_names[i * maps_per_column:(i + 1) * maps_per_column]:
                    st.code(map_name, language=None)
    return all_maps


def style_changed_cells(new_df: pd.DataFrame, old_df: pd.DataFrame):
    """Compares two DataFrames and returns a Styler object with changed cells highlighted."""
    try:
        new_df_c = new_df.copy().astype(float)
        old_df_c = old_df.copy().astype(float)
        old_df_aligned, new_df_aligned = old_df_c.align(new_df_c, join='outer', axis=None)
        style_df = pd.DataFrame('', index=new_df.index, columns=new_df.columns)
        increase_style = 'background-color: #2B442B'
        decrease_style = 'background-color: #442B2B'
        style_df[new_df_aligned > old_df_aligned] = increase_style
        style_df[new_df_aligned < old_df_aligned] = decrease_style
        return new_df.style.apply(lambda x: style_df, axis=None).format("{:.2f}")
    except (ValueError, TypeError):
        st.warning("Could not apply cell highlighting due to a data type mismatch. Displaying unstyled table.")
        return new_df.style.format("{:.2f}")


# These analysis functions are cached. The global clear on "Run" will manage them.
@st.cache_data(show_spinner="Running WG analysis...")
def cached_run_wg_analysis(*args, **kwargs):
    return run_wg_analysis(*args, **kwargs)


@st.cache_data(show_spinner="Running MFF analysis...")
def cached_run_mff_analysis(*args, **kwargs):
    return run_mff_analysis(*args, **kwargs)


@st.cache_data(show_spinner="Running KNK analysis...")
def cached_run_knk_analysis(*args, **kwargs):
    return run_knk_analysis(*args, **kwargs)


# --- UI LAYOUT ---

# --- 1. Sidebar ---
with st.sidebar:
    st.image("yaktune-website-favicon-black.png", use_container_width='auto')
    st.header("⚙️ Tuner Settings")

    # --- Module Selection ---
    run_wg = st.checkbox("Tune Wastegate (WG)", value=True, key="run_wg")
    run_mff = st.checkbox("Tune Mass Fuel Flow (MFF)", value=True, key="run_mff")
    run_ign = st.checkbox("Tune Ignition (KNK)", value=False, key="run_ign")
    st.divider()

    # --- Firmware Selection ---
    st.subheader("Firmware")
    # This placeholder will be populated by the main script logic
    firmware_placeholder = st.empty()
    st.divider()

    # --- Global & Module-Specific Settings ---
    st.subheader("Global Settings")
    oil_temp_unit = st.radio(
        "Oil Temperature Unit in Log File", ('F', 'C'), index=0, horizontal=True
    )
    st.divider()

    if run_wg:
        st.subheader("WG Settings")
        use_swg_logic = st.checkbox("Use Custom WGDC Logic", key="use_swg_logic")

    if run_ign:
        st.subheader("Ignition Settings")
        max_adv = st.slider("Max Advance", 0.0, 2.0, 0.75, 0.25, key="max_adv")

    st.divider()
    # --- Donation Link ---
    paypal_link = "https://www.paypal.com/donate/?hosted_button_id=MN43RKBR8AT6L"
    st.markdown(f"""
    <style>
        .paypal-button {{
            display: inline-block; padding: 8px 16px; font-size: 14px; font-weight: bold;
            color: #ffffff !important; background-color: #0070ba; border: none; border-radius: 5px;
            text-align: center; text-decoration: none; cursor: pointer; transition: background-color 0.3s;
        }}
        .paypal-button:hover {{
            background-color: #005ea6; color: #ffffff !important; text-decoration: none;
        }}
    </style>
    <div style="text-align: center; margin-top: 20px;">
        <a href="{paypal_link}" target="_blank" class="paypal-button">☕ Support YAKtuner</a>
    </div>
    """, unsafe_allow_html=True)

# --- 2. Main Panel ---
st.subheader("1. Upload Tune & Log Files")
uploaded_bin_file = st.file_uploader("Upload .bin file", type=['bin', 'all'])
uploaded_log_files = st.file_uploader("Upload .csv log files", type=['csv'], accept_multiple_files=True)

# --- Dynamic Firmware Logic ---
if 'firmware_id' not in st.session_state:
    st.session_state.firmware_id = None

with firmware_placeholder.container():
    detected_fw = None
    if uploaded_log_files:
        detected_fw = get_firmware_from_log(uploaded_log_files[0])

    manual_selection_active = False
    if uploaded_log_files:
        manual_selection_active = st.checkbox("Manually Select Firmware", key="manual_fw_selection")

    if manual_selection_active:
        with st.spinner("Fetching available firmwares..."):
            # This now returns a map: {'fw_id': 'path/to/fw.xdf'}
            available_firmwares_map = get_available_firmwares_from_github()
        if available_firmwares_map:
            # We display the keys (firmware IDs) to the user
            firmware_ids = sorted(available_firmwares_map.keys())
            try:
                # Default the selectbox to the currently active firmware
                current_index = firmware_ids.index(st.session_state.firmware_id)
            except (ValueError, TypeError):
                current_index = 0

            selected_fw = st.selectbox(
                "Select Firmware",
                options=firmware_ids,
                index=current_index
            )
            st.session_state.firmware_id = selected_fw
        else:
            st.warning("Could not fetch firmware list. Manual selection unavailable.")
    else:
        if detected_fw:
            st.session_state.firmware_id = detected_fw

    active_fw = st.session_state.get('firmware_id')
    if active_fw:
        st.info(f"**Active Firmware:**\n`{active_fw}`")
    else:
        st.info("**Firmware:**\n`Upload log to detect`")

# --- Pre-fetch XDF content and path ---
xdf_content = None
xdf_path = None
firmware = st.session_state.get('firmware_id')
if firmware:
    # We need the map to find the path
    firmware_map = get_available_firmwares_from_github()
    xdf_path = firmware_map.get(firmware)
    if xdf_path:
        # This spinner provides immediate feedback when the firmware ID changes.
        # The result is cached, so it's instant on subsequent reruns unless the ID changes.
        with st.spinner(f"Loading XDF for firmware {firmware}..."):
            xdf_content = download_xdf(xdf_path)

# --- 3. Run Button and Analysis Logic ---
st.divider()

if st.button("🚀 Run YAKtuner Analysis", type="primary", use_container_width=True):
    # --- The Sledgehammer Fix ---
    # This programmatically mimics the manual cache clear that you proved works.
    # It ensures that all cached analysis functions are forced to re-run.
    st.cache_data.clear()
    # --- End Fix ---

    st.session_state.run_analysis = True
    for key in ['mapping_initialized', 'mapping_complete', 'vars_to_map', 'varconv_array', 'log_df_mapped']:
        if key in st.session_state:
            del st.session_state[key]

if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    firmware = st.session_state.get('firmware_id')  # Re-fetch in case it changed
    if not uploaded_bin_file or not uploaded_log_files or not firmware:
        missing = []
        if not uploaded_bin_file: missing.append("BIN file")
        if not uploaded_log_files: missing.append("Log file(s)")
        if not firmware: missing.append("Firmware (upload log or select manually)")
        st.error(f"Please provide all required inputs. Missing: {', '.join(missing)}")
        st.session_state.run_analysis = False
        st.stop()

    # Re-download the XDF content within the run block to be absolutely sure it's fresh.
    # The download_xdf function is cached, so this is an instant operation.
    firmware_map = get_available_firmwares_from_github()
    xdf_path = firmware_map.get(firmware)
    xdf_content = download_xdf(xdf_path)

    if xdf_content is None:
        st.error(f"Failed to load XDF for firmware {firmware}. Cannot proceed.")
        st.session_state.run_analysis = False
        st.stop()

    try:
        xdf_name = os.path.basename(xdf_path) if xdf_path else f"{firmware}.xdf"
        wg_results, mff_results, knk_results = None, None, None
        all_maps_data = {}

        log_df = pd.concat(
            (pd.read_csv(f, encoding='latin1', skiprows=LOG_METADATA_ROWS_TO_SKIP).iloc[:, :-1] for f in
             uploaded_log_files),
            ignore_index=True
        )

        if not os.path.exists(default_vars):
            raise FileNotFoundError(f"Critical file missing: '{default_vars}'")

        logvars_df = pd.read_csv(default_vars, header=None)
        mapped_log_df = map_log_variables_streamlit(log_df, logvars_df)

        if mapped_log_df is not None:
            # --- FIX: Perform unit conversion AFTER variable mapping ---
            # This ensures the conversion is applied to the standardized 'OILTEMP' column.
            if 'OILTEMP' in mapped_log_df.columns and oil_temp_unit == 'C':
                mapped_log_df['OILTEMP'] = mapped_log_df['OILTEMP'] * 1.8 + 32
                st.toast("Oil Temperature converted to Fahrenheit.", icon="🌡️")
            # --- END FIX ---

            with st.status("Starting YAKtuner analysis...", expanded=True) as status:
                if 'updated_varconv_df' in st.session_state:
                    with st.expander("View Variable Mapping Results"):
                        varconv_array = st.session_state.updated_varconv_df.to_numpy()
                        mapping_summary_df = pd.DataFrame({
                            "Required Variable": varconv_array[2, 1:] if varconv_array.shape[0] > 2 else varconv_array[
                                                                                                         1, 1:],
                            "Matched Log Column": varconv_array[0, 1:],
                            "Internal App Name": varconv_array[1, 1:]
                        })
                        st.dataframe(mapping_summary_df, use_container_width=True)

                status.update(label="Loading tune files...")
                bin_content = uploaded_bin_file.getvalue()

                # This call now uses the fresh xdf_content from the main script body
                all_maps = load_all_maps_streamlit(
                    bin_content=bin_content,
                    xdf_content=xdf_content,
                    xdf_name=xdf_name
                )

                if all_maps:
                    log_for_mff = mapped_log_df.copy()

                    if run_wg:
                        with st.status("Running Wastegate (WG) analysis...", expanded=True) as module_status:
                            try:
                                x_axis_key, y_axis_key, main_table_key = (
                                    'wgdc_cust_X', 'wgdc_cust_Y', 'wgdc_cust') if use_swg_logic else (
                                    'wgdc_X', 'wgdc_Y', 'wgdc')
                                essential_keys = [x_axis_key, y_axis_key, main_table_key]
                                module_maps = {key: all_maps.get(key) for key in essential_keys}
                                if any(v is None for v in module_maps.values()):
                                    raise KeyError(
                                        f"A required map for WG tuning is missing: {[k for k, v in module_maps.items() if v is None]}")
                                all_maps_data['wg'] = module_maps

                                wg_results = cached_run_wg_analysis(
                                    log_df=mapped_log_df, wgxaxis=module_maps[x_axis_key],
                                    wgyaxis=module_maps[y_axis_key],
                                    oldWG=module_maps[main_table_key], logvars=mapped_log_df.columns.tolist(),
                                    WGlogic=use_swg_logic
                                )
                                module_status.update(label="Wastegate (WG) analysis complete.", state="complete",
                                                     expanded=False)
                            except Exception as e:
                                st.error(f"An unexpected error occurred during WG tuning: {e}")
                                module_status.update(label="Wastegate (WG) analysis failed.", state="error",
                                                     expanded=True)

                    if run_mff:
                        with st.status("Running Fuel Factor (MFF) analysis...", expanded=True) as module_status:
                            try:
                                keys = ['MFFtable_X', 'MFFtable_Y', 'MFFtable']
                                module_maps = {key: all_maps.get(key) for key in keys}
                                if any(v is None for v in module_maps.values()):
                                    raise KeyError(
                                        f"A required map for MFF tuning is missing: {[k for k, v in module_maps.items() if v is None]}")
                                all_maps_data['mff'] = module_maps

                                mff_results = cached_run_mff_analysis(
                                    log=log_for_mff, mffxaxis=module_maps['MFFtable_X'],
                                    mffyaxis=module_maps['MFFtable_Y'],
                                    mfftable=module_maps['MFFtable'], logvars=mapped_log_df.columns.tolist()
                                )
                                module_status.update(label="Fuel Factor (MFF) analysis complete.", state="complete",
                                                     expanded=False)
                            except Exception as e:
                                st.error(f"An unexpected error occurred during MFF tuning: {e}")
                                module_status.update(label="Fuel Factor (MFF) analysis failed.", state="error",
                                                     expanded=True)

                    if run_ign:
                        with st.status("Running Ignition (KNK) analysis...", expanded=True) as module_status:
                            try:
                                keys = ['igxaxis', 'igyaxis']
                                module_maps = {key: all_maps.get(key) for key in keys}
                                if any(v is None for v in module_maps.values()):
                                    raise KeyError(
                                        f"A required map for KNK tuning is missing: {[k for k, v in module_maps.items() if v is None]}")
                                all_maps_data['knk'] = module_maps

                                knk_results = cached_run_knk_analysis(
                                    log=mapped_log_df, igxaxis=module_maps['igxaxis'], igyaxis=module_maps['igyaxis'],
                                    max_adv=max_adv
                                )
                                module_status.update(label="Ignition (KNK) analysis complete.", state="complete",
                                                     expanded=False)
                            except Exception as e:
                                st.error(f"An unexpected error occurred during KNK tuning: {e}")
                                module_status.update(label="Ignition (KNK) analysis failed.", state="error",
                                                     expanded=True)

                status.update(label="Analysis complete!", state="complete", expanded=False)
                st.balloons()

            # --- Phase 3: Display All Results ---
            st.header("📈 Analysis Results")

            if wg_results and wg_results.get('status') == 'Success':
                with st.expander("Wastegate (WG) Tuning Results", expanded=True):
                    if wg_results['warnings']:
                        for warning in wg_results['warnings']: st.warning(f"WG Analysis Warning: {warning}")
                    recommended_wg_df, scatter_plot = wg_results['results_wg'], wg_results['scatter_plot_fig']
                    module_maps = all_maps_data['wg']
                    x_axis_key, y_axis_key, main_table_key = (
                        'wgdc_cust_X', 'wgdc_cust_Y', 'wgdc_cust') if use_swg_logic else ('wgdc_X', 'wgdc_Y', 'wgdc')
                    exh_labels = [str(x) for x in module_maps[x_axis_key]]
                    int_labels = [str(y) for y in module_maps[y_axis_key]]
                    original_wg_df = pd.DataFrame(module_maps[main_table_key], index=int_labels, columns=exh_labels)
                    styled_wg_table = style_changed_cells(recommended_wg_df, original_wg_df)
                    tab1, tab2 = st.tabs(["📈 Recommended Table", "📊 Scatter Plot"])
                    with tab1:
                        display_table_with_copy_button("#### Recommended WGDC Base Table", styled_wg_table,
                                                       recommended_wg_df)
                    with tab2:
                        if scatter_plot:
                            st.pyplot(scatter_plot)
                        else:
                            st.info("Scatter plot was not generated.")

            if mff_results:
                with st.expander("Multiplicative Fuel Factor (MFF) Tuning Results", expanded=True):
                    if mff_results['warnings']:
                        for warning in mff_results['warnings']: st.warning(f"MFF Analysis Warning: {warning}")

                    if mff_results.get('status') == 'Success':
                        recommended_mff_df = mff_results['results_mff']
                        module_maps = all_maps_data['mff']
                        original_df = pd.DataFrame(module_maps['MFFtable'],
                                                   index=[str(y) for y in module_maps['MFFtable_Y']],
                                                   columns=[str(x) for x in module_maps['MFFtable_X']])
                        styled_table = style_changed_cells(recommended_mff_df, original_df)
                        display_table_with_copy_button(f"#### Recommended Fuel Scalar Table multiplier", styled_table,
                                                       recommended_mff_df)

            if knk_results and knk_results.get('status') == 'Success':
                with st.expander("Ignition Timing (KNK) Tuning Results", expanded=True):
                    if knk_results['warnings']:
                        for warning in knk_results['warnings']: st.warning(f"KNK Analysis Warning: {warning}")
                    recommended_knk_df, scatter_plot = knk_results['results_knk'], knk_results['scatter_plot_fig']
                    tab1, tab2 = st.tabs(["📈 Recommended Correction Table", "📊 Knock Scatter Plot"])
                    with tab1:
                        styled_table = recommended_knk_df.style.format("{:.2f}").background_gradient(cmap='viridis',
                                                                                                     axis=None)
                        display_table_with_copy_button("#### Recommended Ignition Correction Table", styled_table,
                                                       recommended_knk_df)
                    with tab2:
                        if scatter_plot:
                            st.pyplot(scatter_plot)
                        else:
                            st.info("Scatter plot was not generated (no knock events found).")

            st.session_state.run_analysis = False

    except Exception as e:
        st.error(f"An unexpected error occurred during the analysis: {e}")
        st.write("You can help improve YAKtuner by sending this error report to the developer.")
        traceback_str = traceback.format_exc()
        with st.form(key="error_report_form"):
            st.write("**An unexpected error occurred.** You can help by sending this report.")
            user_description = st.text_area("Optional: Please describe what you were doing when the error occurred.")
            user_contact = st.text_input("Optional: Email or username for follow-up questions.")
            st.text_area("Technical Error Details (for submission)", value=traceback_str, height=200, disabled=True)
            submit_button = st.form_submit_button("Submit Error Report")
            if submit_button:
                with st.spinner("Sending report..."):
                    success, message = send_to_google_sheets(traceback_str, user_description, user_contact)
                    if success:
                        st.success("Thank you! Your error report has been sent.")
                    else:
                        st.error(f"Sorry, the report could not be sent. Reason: {message}")
        with st.expander("Click to view technical error details"):
            st.code(traceback_str, language=None)
        st.session_state.run_analysis = False