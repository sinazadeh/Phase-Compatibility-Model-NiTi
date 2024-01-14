# -------------------------------------
# Code by Sina Zadeh
# Nobember 2023
# https://www.sina.science/
# -------------------------------------

import pandas as pd
from CBFV import composition
from functools import reduce
from HEACalculator import HEACalculator
import numpy as np
import math
import contextlib
import io
import logging


class DataHandler:
    def __init__(self, data):
        self.df = pd.DataFrame(data)

    def generate_dataframe(self):
        return self.df


class LambdaCalculator:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def calculate_lambdas(a0, a, b, c, beta):
        a, b, c = sorted([a, b, c])
        B_matrix = np.array(
            [
                [a / a0, 0, (np.sqrt(2) * c *
                             math.cos(math.radians(beta))) / (2 * a0)],
                [0, (np.sqrt(2) * b) / (2 * a0), 0],
                [0, 0, (np.sqrt(2) * c * math.sin(math.radians(beta))) / (2 * a0)],
            ]
        )
        try:
            _, singular_values, _ = np.linalg.svd(B_matrix, full_matrices=True)
            return np.sort(singular_values)
        except (ValueError, np.linalg.LinAlgError):
            return [np.nan, np.nan, np.nan]

    def generate_lambdas(self):
        results = self.df.apply(
            lambda row: self.calculate_lambdas(
                row["a0 (A)"], row["a (A)"], row["b (A)"], row["c (A)"], row["beta"]
            ),
            axis=1,
            result_type="expand",
        )

        results.columns = [
            "lambda1_calculated",
            "lambda2_calculated",
            "lambda3_calculated",
        ]
        self.df = pd.concat([self.df, results], axis=1)
        return self.df


class Lambda2Model:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def determine_type_and_lambda(x):
        if x < 0.7:
            type_value = "B19'"
            central_lambda = 1.0333 * x + 0.2512
        elif x >= 0.7:
            type_value = "B19"
            central_lambda = 1.0333 * x + 0.2752

        return type_value, round(central_lambda, 3)

    def predict_transformation_and_lambda2(self):
        type_lambda_df = self.df["jarvis_avg_first_ion_en_divi_voro_coord"].apply(
            self.determine_type_and_lambda
        )

        self.df["Predicted_Transformation_Type"] = type_lambda_df.apply(
            lambda x: x[0])
        self.df["Predicted_Lambda2"] = type_lambda_df.apply(lambda x: x[1])
        try:
            self.df = self.df[
                [
                    "composition",
                    "Alloy_System",
                    "jarvis_avg_first_ion_en_divi_voro_coord",
                    "Predicted_Transformation_Type",
                    "Predicted_Lambda2",
                ]
            ]
        except:
            pass
        return self.df


class FeatureGenerator:
    replacements = {
        "CuMoNiTi": "NiTiCuMo",
        "CuHfNiPdTiZr": "NiTiCuHfPdZr",
        "PdTiVZr": "PdTiZrV",
        "AlCuNiTi": "NiTiAlCu",
        "CuNiPtTi": "NiTiCuPt",
        "AlNiTiZr": "NiTiZrAl",
        "CuNiSiTi": "NiTiCuSi",
        "HfNiSnTi": "NiTiHfSn",
        "NiPbTiZr": "NiTiPbZr",
        "CuHfNiPbTi": "NiTiHfCuPb",
        "CoNiPbTi": "NiTiCoPb",
        "CuNiPbTiZr": "NiTiCuZrPb",
        "CuHfNiPbTiZr": "NiTiCuHfPbZr",
        "AuCuNiTi": "NiTiCuAu",
        "CuNiPdTi": "NiTiPdCu",
        "NiPdPtTi": "NiTiPdPt",
        "NbNiTiZr": "NiTiNbZr",
        "HfNiTiZr": "NiTiHfZr",
        "NiPdTaTi": "NiTiPdTa",
        "CuHfNiTi": "NiTiHfCu",
        "CuHfNiTiZr": "NiTiHfZrCu",
        "BNiPdTi": "NiTiPdB",
        "BNiTiZr": "NiTiZrB",
        "CoCuNiTi": "NiTiCuCo",
        "CoInMnNi": "NiMnCoIn",
        "CoNiPdTi": "NiTiPdCo",
        "CuFeHfNiTi": "NiTiHfFeCu",
        "NiPdScTi": "NiTiPdSc",
        "HfNiTaTi": "NiTiHfTa",
        "CuNbNiTi": "NiTiCuNb",
        "CuNiTiZr": "NiTiCuZr",
        "HfNiPdTi": "NiTiPdHf",
        "CuNiTi": "NiTiCu",
        "NiPtTi": "NiTiPt",
        "HfNiTi": "NiTiHf",
        "NbNiTi": "NiTiNb",
        "NiPdTi": "NiTiPd",
        "NiTaTi": "NiTiTa",
        "NiReTi": "NiTiRe",
        "NiSiTi": "NiTiSi",
        "NiSnTi": "NiTiSn",
        "NiSbTi": "NiTiSb",
        "NiScTi": "NiTiSc",
        "NiTeTi": "NiTiTe",
        "NiPbTi": "NiTiPb",
        "NiPrTi": "NiTiPr",
        "AuNiTi": "NiTiAu",
        "NdNiTi": "NiTiNd",
        "NiRhTi": "NiTiRh",
    }

    def __init__(self, df):
        self.df = df.copy()
        self.ext_df = df.copy()
        self.columns_range = self.df.columns
        self.base_features_generated = False

    @staticmethod
    def stringify(x):
        return str(float(x)) if pd.notnull(x) and x != 0 else ""

    @staticmethod
    def is_almost_zero(num):
        return abs(num) < 1e-9

    @staticmethod
    def greatest_common_divisor(a, b):
        return (
            a
            if FeatureGenerator.is_almost_zero(b)
            else FeatureGenerator.greatest_common_divisor(b, a % b)
        )

    def gcd_of_array(self, array):
        return reduce(self.greatest_common_divisor, array)

    def correct_ratios(self, arr_values):
        gcd = self.gcd_of_array(arr_values)
        return [round(a / gcd) for a in arr_values]

    def to_formula_string(self, dct):
        corrected_vals = self.correct_ratios(list(dct.values()))
        elements = [
            element for element, value in zip(dct.keys(), corrected_vals) if value != 0
        ]
        values = [value for value in corrected_vals if value != 0]
        return "".join(
            [f"{element}{value}" for element, value in zip(elements, values)]
        )

    def generate_composition_formula(self):
        self.df["composition"] = (
            self.df[self.columns_range]
            .apply(lambda x: x.map(self.stringify))
            .apply(
                lambda row: "".join(
                    [f"{k}{v}" for k, v in row.items() if v != "" and v != "0.0"]
                ),
                axis=1,
            )
        )

        dict_alloy_compositions = self.df[self.columns_range].to_dict("split")
        formula_list = [
            self.to_formula_string(
                dict(zip(dict_alloy_compositions["columns"], row_data))
            )
            for row_data in dict_alloy_compositions["data"]
        ]
        self.df["formula"] = formula_list
        return self.df

    def generate_features(self):
        self.df["Alloy_System"] = (
            self.df["formula"]
            .replace("\d+", "", regex=True)
            .replace(FeatureGenerator.replacements, regex=True)
        )
        self.df["niti_base"] = self.df.Alloy_System.apply(
            lambda x: "True" if "NiTi" in x else "False"
        )
        temp_df = self.df.copy()

        # Generate HEA related features
        lst = []
        for alloy in temp_df["composition"]:
            lst.append(HEACalculator(alloy, csv=True).get_csv_list())
        headers = [
            "Formula",
            "Density",
            "Delta",
            "Omega",
            "Gamma",
            "lambda",
            "VEC",
            "Mixing Enthalpy",
            "Mixing Entropy",
            "Melting Temperature",
        ]
        hea_features_df = pd.DataFrame(lst, columns=headers).iloc[:, 1:]

        # Generate compositional features
        self.df["target"] = 0
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            logging.getLogger().setLevel(logging.CRITICAL)

            X_jarvis, _, _, _ = composition.generate_features(
                self.df, elem_prop="jarvis", sum_feat=True
            )
            X_magpie, _, _, _ = composition.generate_features(
                self.df, elem_prop="magpie", sum_feat=True
            )
            X_oliynyk, _, _, _ = composition.generate_features(
                self.df, elem_prop="oliynyk", sum_feat=True
            )
            X_mat2vec, _, _, _ = composition.generate_features(
                self.df, elem_prop="mat2vec", sum_feat=True
            )
            X_onehot, _, _, _ = composition.generate_features(
                self.df, elem_prop="onehot", sum_feat=True
            )
        self.df = pd.concat(
            [
                self.ext_df,
                X_jarvis.add_prefix("jarvis_"),
                X_magpie.add_prefix("magpie_"),
                X_oliynyk.add_prefix("oliynyk_"),
                X_mat2vec.add_prefix("mat2vec_"),
                X_onehot.add_prefix("onehot_"),
                hea_features_df.add_prefix("hea_"),
                temp_df["Alloy_System"],
                temp_df["composition"],
                temp_df["formula"],
            ],
            axis=1,
        )
        self.df = self.df[
            [
                "composition",
                "formula",
                "Alloy_System",
                "jarvis_avg_first_ion_en_divi_voro_coord",
                "jarvis_dev_mol_vol",
                "hea_Delta",
                "jarvis_avg_first_ion_en",
                "jarvis_avg_voro_coord",
                "jarvis_avg_atom_mass",
                "jarvis_avg_mol_vol",
            ]
        ]
        return self.df

    def generate_features_all(self):
        # Generate alloy system
        self.df["Alloy_System"] = (
            self.df["formula"]
            .replace("\d+", "", regex=True)
            .replace(FeatureGenerator.replacements, regex=True)
        )
        self.df["niti_base"] = self.df.Alloy_System.apply(
            lambda x: "True" if "NiTi" in x else "False"
        )
        temp_df = self.df.copy()

        # Generate HEA related features
        lst = []
        for alloy in temp_df["composition"]:
            lst.append(HEACalculator(alloy, csv=True).get_csv_list())
        headers = [
            "Formula",
            "Density",
            "Delta",
            "Omega",
            "Gamma",
            "lambda",
            "VEC",
            "Mixing Enthalpy",
            "Mixing Entropy",
            "Melting Temperature",
        ]
        hea_features_df = pd.DataFrame(lst, columns=headers).iloc[:, 1:]

        # Generate compositional features
        self.df["target"] = 0

        X_jarvis, _, _, _ = composition.generate_features(
            self.df, elem_prop="jarvis", sum_feat=True
        )
        X_magpie, _, _, _ = composition.generate_features(
            self.df, elem_prop="magpie", sum_feat=True
        )
        X_oliynyk, _, _, _ = composition.generate_features(
            self.df, elem_prop="oliynyk", sum_feat=True
        )
        X_mat2vec, _, _, _ = composition.generate_features(
            self.df, elem_prop="mat2vec", sum_feat=True
        )
        X_onehot, _, _, _ = composition.generate_features(
            self.df, elem_prop="onehot", sum_feat=True
        )
        self.df = pd.concat(
            [
                self.ext_df,
                X_jarvis.add_prefix("jarvis_"),
                X_magpie.add_prefix("magpie_"),
                X_oliynyk.add_prefix("oliynyk_"),
                X_mat2vec.add_prefix("mat2vec_"),
                X_onehot.add_prefix("onehot_"),
                hea_features_df.add_prefix("hea_"),
                temp_df["Alloy_System"],
            ],
            axis=1,
        )
        return self.df
