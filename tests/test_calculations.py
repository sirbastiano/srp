from calculations import get_public_orgs, revenue_per_industry
import pytest


def test_get_public_orgs():
    expected_num_public_orgs = (
        54  # replace with the actual number of active drivers in your dataset
    )
    assert get_public_orgs() == expected_num_public_orgs


@pytest.mark.parametrize(
    "Industry, expected_revenue",
    [
        ("Wireless", 3560000.0),
        ("Textiles", 4285000.0),
        ("Accounting", 2930000.0),
    ],
    ids=["Wireless", "Textiles", "Accounting"],
)
def test_revenue_per_industry(Industry, expected_revenue):
    revenue_ratio = revenue_per_industry()
    assert (
        revenue_ratio[Industry] == expected_revenue
    ), f"expected: {expected_revenue}, got: {revenue_ratio[Industry]}"
