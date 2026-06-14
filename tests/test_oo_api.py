"""
Tests for the v3.1 object-oriented API (odc.model.ODCModel) and the dict-compatible
result objects (odc.results). These guard the refactor: the OO path must be numerically
identical to the legacy functional shims, and the result objects must remain drop-in
replacements for the dicts that build_dashboard.py / build_notebook.py / the old tests use.
"""
from dataclasses import asdict, fields

from odc import core
from odc.core import (P, ODCModel, Spacecraft, power_thermal_mass, lcoc_and_npv,
                      PowerThermalMass, LCOCResult)
from odc.scenarios import TODAY, MATURE, SA26
from odc.orbits import EQUATORIAL_LEO, DDSS
from odc.sizes import STATION_16MW
from odc.schema import GROUPS, slider_fields, _p_field_names


# ---------------- OO == functional (exact numerical identity) ----------------

def test_oo_evaluate_matches_functional_shim_exactly():
    for sc in (TODAY, MATURE, SA26):
        a = ODCModel.from_scenario(sc).evaluate()
        b = lcoc_and_npv(sc)
        assert a.as_dict() == b.as_dict()
        assert a.lcoc_s == b["lcoc_s"]      # exact, both go through ODCModel


def test_spacecraft_is_odcmodel_alias():
    assert Spacecraft is ODCModel


# ---------------- result objects behave like the legacy dicts ----------------

def test_attribute_and_item_access_agree():
    r = lcoc_and_npv(SA26)
    for f in fields(LCOCResult):
        assert r[f.name] == getattr(r, f.name)
    # nested
    assert r.pt.A_rad == r["pt"]["A_rad"]
    assert r.sa.gpu_hr_s == r["sa"]["gpu_hr_s"]


def test_result_is_mapping_compatible():
    r = lcoc_and_npv(SA26)
    assert "lcoc_s" in r and "not_a_field" not in r
    assert r.get("not_a_field", 123) == 123
    d = dict(r)                                  # uses keys() + __getitem__
    assert set(d) == {f.name for f in fields(LCOCResult)}
    assert {**r}["ratio"] == r.ratio             # ** unpacking


def test_pt_as_dict_equals_legacy_dict_shape():
    pt = power_thermal_mass(SA26)
    assert isinstance(pt, PowerThermalMass)
    d = pt.as_dict()
    assert isinstance(d, dict) and d["A_rad"] == pt.A_rad
    # deep nested shape on the full result
    r = lcoc_and_npv(SA26).as_dict()
    assert isinstance(r["pt"], dict) and r["pt"]["M_dry"] == lcoc_and_npv(SA26).pt.M_dry


# ---------------- fluent builders ----------------

def test_with_orbit_reproduces_orbit_apply():
    a = ODCModel(TODAY).with_orbit(EQUATORIAL_LEO).power_thermal_mass().as_dict()
    b = power_thermal_mass(EQUATORIAL_LEO.apply(TODAY)).as_dict()
    assert a == b


def test_ddss_apply_is_backward_compatible_via_oo():
    base = ODCModel(TODAY).power_thermal_mass().M_dry
    ddss = ODCModel(TODAY).with_orbit(DDSS).power_thermal_mass().M_dry
    assert abs(base - ddss) / base < 0.01


def test_with_size_sets_power_basis():
    m = ODCModel(MATURE).with_size(STATION_16MW)
    assert m.p_it_mw == STATION_16MW.it_mw
    assert m.evaluate().as_dict() == lcoc_and_npv(MATURE, P_it_MW=STATION_16MW.it_mw).as_dict()


def test_replace_overrides_params_immutably():
    base = ODCModel(TODAY)
    cheaper = base.replace(launch_kg=300)
    assert cheaper.p.launch_kg == 300
    assert base.p.launch_kg == 1600.0           # original untouched
    assert cheaper.evaluate().lcoc_s < base.evaluate().lcoc_s


# ---------------- export / schema contracts ----------------

def test_asdict_of_P_is_flat_and_covers_slider_fields():
    d = asdict(P())                              # dashboard export contract
    assert all(not isinstance(v, dict) for v in d.values())   # flat, no nesting
    assert set(slider_fields()) <= set(d)        # every slider maps to a real P field


def test_every_schema_field_exists_on_P():
    assert set(slider_fields()) <= _p_field_names()


def test_schema_groups_well_formed():
    for group_label, rows in GROUPS:
        assert isinstance(group_label, str)
        for row in rows:
            assert len(row) == 8                 # field,label,unit,lo,hi,step,kind,note
            field, label, unit, lo, hi, step, kind, note = row
            assert lo < hi and kind in (None, "log", "pct")
