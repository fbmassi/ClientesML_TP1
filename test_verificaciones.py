"""Verificaciones pequeñas de datos, aislamiento, métricas y evidencia ejecutada."""
import json
import unittest

import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from experimento import (ROOT, SEED, NUMERICAS, cargar_datos, separar,
                         preprocesador, metricas)


class VerificacionesTP1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.datos = cargar_datos()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = separar(cls.datos)

    def test_original_conversion_y_objetivo(self):
        self.assertEqual(self.raw.shape, (7043, 21))
        self.assertEqual(int(self.datos.TotalCharges.isna().sum()), 11)
        self.assertTrue(self.datos.loc[self.datos.TotalCharges.isna(), "tenure"].eq(0).all())
        self.assertFalse(self.raw.customerID.duplicated().any())
        self.assertEqual(self.datos.Churn.value_counts().to_dict(), {0: 5174, 1: 1869})
        self.assertNotIn("customerID", self.X_train)
        self.assertNotIn("Churn", self.X_train)

    def test_particiones_y_folds(self):
        train, test = set(self.X_train.index), set(self.X_test.index)
        self.assertEqual((len(train), len(test)), (5634, 1409))
        self.assertFalse(train & test)
        self.assertEqual(train | test, set(self.datos.index))
        self.assertLess(abs(self.y_train.mean()-self.y_test.mean()), .002)
        usados = []
        cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for a, b in cv.split(self.X_train, self.y_train):
            ai, bi = set(self.X_train.iloc[a].index), set(self.X_train.iloc[b].index)
            self.assertFalse(ai & bi)
            self.assertFalse((ai | bi) & test)
            self.assertEqual(ai | bi, train)
            usados.extend(bi)
        self.assertEqual(len(usados), len(set(usados)))
        self.assertEqual(set(usados), train)

    def test_transformaciones_solo_aprenden_train(self):
        train = self.X_train.copy()
        test = self.X_test.iloc[:3].copy()
        test["MonthlyCharges"] = 1e9  # Nunca debe influir en los parámetros aprendidos.
        test["Contract"] = "categoria_no_vista"
        test["TotalCharges"] = np.nan
        p = preprocesador()
        p.fit(train)
        imp = p.named_transformers_["numericas"].named_steps["imputar"]
        escala = p.named_transformers_["numericas"].named_steps["escalar"]
        np.testing.assert_allclose(imp.statistics_, train[NUMERICAS].median().to_numpy())
        np.testing.assert_allclose(escala.mean_, train[NUMERICAS].fillna(train[NUMERICAS].median()).mean().to_numpy())
        antes = escala.mean_.copy()
        transformado = p.transform(test)
        self.assertTrue(np.isfinite(transformado).all())
        np.testing.assert_array_equal(escala.mean_, antes)
        categorias = p.named_transformers_["categoricas"].named_steps["codificar"].categories_
        self.assertFalse(any("categoria_no_vista" in c for c in categorias))

    def test_metricas_guardadas_recalculadas(self):
        resultados = ROOT / "resultados"
        prueba = pd.read_csv(resultados / "prueba.csv").set_index("modelo")
        pred = pd.read_csv(resultados / "predicciones_prueba.csv")
        np.testing.assert_array_equal(pred.fila_original, self.X_test.index)
        np.testing.assert_array_equal(pred.real, self.y_test)
        meta = json.loads((resultados / "experimento.json").read_text(encoding="utf-8"))
        for nombre in prueba.index:
            prob = pred[f"{nombre}_prob"]
            calculadas = (prob >= .5).astype(int)
            np.testing.assert_array_equal(calculadas, pred[f"{nombre}_pred"])
            for k, v in metricas(pred.real, calculadas, prob).items():
                self.assertAlmostEqual(v, prueba.loc[nombre, k], places=12)
            self.assertEqual(np.asarray(meta["matrices"][nombre]).sum(), len(self.X_test))

    def test_seleccion_es_maximo_cv(self):
        r = ROOT / "resultados"
        cv = pd.read_csv(r / "validacion.csv")
        resumen = pd.read_csv(r / "seleccion.csv")
        meta = json.loads((r / "experimento.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cv), 8)
        self.assertEqual(meta["seleccionado_por_cv"], resumen.iloc[0].modelo)
        self.assertAlmostEqual(resumen.iloc[0].cv_f1_macro, cv.cv_f1_macro.max())
        for _, row in cv.iterrows():
            scores = row[[f"fold_{i}_f1_macro" for i in [1, 2, 3]]].astype(float)
            self.assertAlmostEqual(row.cv_f1_macro, scores.mean(), places=12)
        for red in meta["redes"].values():
            self.assertEqual(red["epocas"], len(red["curva_perdida"]))
            self.assertTrue(np.isfinite(red["curva_perdida"]).all())
        for figura in ["eda", "comparacion", "confusion", "perdida_mlp"]:
            self.assertGreater((r / f"{figura}.png").stat().st_size, 5000)

    def test_notebook_ejecutada_completa(self):
        nb = nbformat.read(ROOT / "TP1_Cliente.ipynb", as_version=4)
        nbformat.validate(nb)
        celdas = [c for c in nb.cells if c.cell_type == "code"]
        self.assertGreaterEqual(len(celdas), 7)
        self.assertEqual([c.execution_count for c in celdas], list(range(1, len(celdas)+1)))
        self.assertFalse(any(o.output_type == "error" for c in celdas for o in c.outputs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
