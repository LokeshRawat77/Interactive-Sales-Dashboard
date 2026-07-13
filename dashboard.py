from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns


# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "sales_data.csv"
VISUALIZATION_DIR = BASE_DIR / "visualizations"
DASHBOARD_DIR = BASE_DIR / "dashboard"
REPORT_DIR = BASE_DIR / "report"

VISUALIZATION_DIR.mkdir(exist_ok=True)
DASHBOARD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load and validate dataset
# --------------------------------------------------
def load_data(file_path: Path) -> pd.DataFrame:
    """Load the sales dataset and validate required columns."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}\n"
            "Place sales_data.csv inside the data folder."
        )

    data = pd.read_csv(file_path)

    required_columns = {
        "Date",
        "Product",
        "Quantity",
        "Price",
        "Customer_ID",
        "Region",
        "Total_Sales",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(sorted(missing_columns))
        )

    return data


# --------------------------------------------------
# Clean and prepare dataset
# --------------------------------------------------
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean text, numeric and date columns."""

    cleaned = data.copy()

    # Remove completely duplicated rows
    cleaned.drop_duplicates(inplace=True)

    # Clean text columns
    for column in ["Product", "Customer_ID", "Region"]:
        cleaned[column] = (
            cleaned[column]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Convert numeric columns
    numeric_columns = ["Quantity", "Price", "Total_Sales"]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # Fill Quantity and Price using median values
    cleaned["Quantity"] = cleaned["Quantity"].fillna(
        cleaned["Quantity"].median()
    )
    cleaned["Price"] = cleaned["Price"].fillna(
        cleaned["Price"].median()
    )

    # Calculate missing sales values
    calculated_sales = cleaned["Quantity"] * cleaned["Price"]
    cleaned["Total_Sales"] = cleaned["Total_Sales"].fillna(
        calculated_sales
    )

    # Convert and validate dates
    cleaned["Date"] = pd.to_datetime(
        cleaned["Date"],
        errors="coerce",
    )

    cleaned.dropna(subset=["Date"], inplace=True)

    # Remove invalid numerical records
    cleaned = cleaned[
        (cleaned["Quantity"] >= 0)
        & (cleaned["Price"] >= 0)
        & (cleaned["Total_Sales"] >= 0)
    ]

    # Create calculated date columns
    cleaned["Year"] = cleaned["Date"].dt.year
    cleaned["Month_Number"] = cleaned["Date"].dt.month
    cleaned["Month"] = cleaned["Date"].dt.month_name()
    cleaned["Year_Month"] = cleaned["Date"].dt.to_period("M").astype(str)

    return cleaned


# --------------------------------------------------
# Prepare analysis tables
# --------------------------------------------------
def prepare_analysis(data: pd.DataFrame) -> dict:
    """Calculate sales, product, region and customer metrics."""

    monthly_sales = (
        data.groupby("Year_Month", as_index=False)
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("Year_Month")
    )

    product_sales = (
        data.groupby("Product", as_index=False)
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("Total_Sales", ascending=False)
    )

    regional_sales = (
        data.groupby("Region", as_index=False)
        .agg(
            Total_Sales=("Total_Sales", "sum"),
            Quantity=("Quantity", "sum"),
        )
        .sort_values("Total_Sales", ascending=False)
    )

    customer_sales = (
        data.groupby("Customer_ID", as_index=False)
        .agg(
            Customer_Value=("Total_Sales", "sum"),
            Orders=("Date", "count"),
            Units=("Quantity", "sum"),
        )
        .sort_values("Customer_Value", ascending=False)
    )

    # Segment customers according to customer lifetime value
    if customer_sales["Customer_Value"].nunique() >= 4:
        customer_sales["Segment"] = pd.qcut(
            customer_sales["Customer_Value"].rank(method="first"),
            q=4,
            labels=["Bronze", "Silver", "Gold", "Platinum"],
        )
    else:
        customer_sales["Segment"] = "Regular"

    segment_summary = (
        customer_sales.groupby("Segment", observed=False, as_index=False)
        .agg(
            Customers=("Customer_ID", "count"),
            Revenue=("Customer_Value", "sum"),
        )
    )

    return {
        "monthly_sales": monthly_sales,
        "product_sales": product_sales,
        "regional_sales": regional_sales,
        "customer_sales": customer_sales,
        "segment_summary": segment_summary,
    }


# --------------------------------------------------
# Create Seaborn visualizations
# --------------------------------------------------
def create_seaborn_visualizations(
    data: pd.DataFrame,
    analysis: dict,
) -> None:
    """Create statistical plots using Seaborn."""

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        palette="deep",
    )

    # 1. Box plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=data,
        x="Product",
        y="Total_Sales",
        hue="Product",
        legend=False,
    )
    plt.title("Sales Distribution by Product")
    plt.xlabel("Product")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(
        VISUALIZATION_DIR / "sales_boxplot.png",
        dpi=300,
    )
    plt.close()

    # 2. Violin plot
    plt.figure(figsize=(11, 6))
    sns.violinplot(
        data=data,
        x="Region",
        y="Total_Sales",
        hue="Region",
        legend=False,
        inner="quartile",
    )
    plt.title("Regional Sales Distribution")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig(
        VISUALIZATION_DIR / "regional_violinplot.png",
        dpi=300,
    )
    plt.close()

    # 3. Correlation heatmap
    numerical_data = data[
        ["Quantity", "Price", "Total_Sales"]
    ]

    correlation = numerical_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        VISUALIZATION_DIR / "correlation_heatmap.png",
        dpi=300,
    )
    plt.close()

    # 4. Professional 2×2 dashboard
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 11),
    )

    monthly = analysis["monthly_sales"]
    products = analysis["product_sales"]
    regions = analysis["regional_sales"]

    sns.lineplot(
        data=monthly,
        x="Year_Month",
        y="Total_Sales",
        marker="o",
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Monthly Sales Trend")
    axes[0, 0].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=products,
        x="Product",
        y="Total_Sales",
        hue="Product",
        legend=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Product Performance")
    axes[0, 1].tick_params(axis="x", rotation=35)

    sns.barplot(
        data=regions,
        x="Region",
        y="Total_Sales",
        hue="Region",
        legend=False,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Regional Sales")

    sns.scatterplot(
        data=data,
        x="Quantity",
        y="Total_Sales",
        hue="Product",
        size="Price",
        sizes=(40, 250),
        alpha=0.75,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Quantity vs Total Sales")

    figure.suptitle(
        "Sales Analytics Dashboard",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        VISUALIZATION_DIR / "seaborn_dashboard.png",
        dpi=300,
    )
    plt.close()


# --------------------------------------------------
# Create interactive Plotly dashboard
# --------------------------------------------------
def create_interactive_dashboard(
    data: pd.DataFrame,
    analysis: dict,
) -> None:
    """Create interactive charts and combine them into one HTML dashboard."""

    monthly = analysis["monthly_sales"]
    products = analysis["product_sales"]
    regions = analysis["regional_sales"]
    customers = analysis["customer_sales"].head(10)
    segments = analysis["segment_summary"]

    # 5. Interactive line chart
    monthly_chart = px.line(
        monthly,
        x="Year_Month",
        y="Total_Sales",
        markers=True,
        title="Monthly Sales Trend",
        hover_data={
            "Year_Month": True,
            "Total_Sales": ":,.2f",
            "Quantity": True,
        },
        template="plotly_white",
    )

    # Dropdown to switch between revenue and quantity
    monthly_chart.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Total Sales",
                        "method": "update",
                        "args": [
                            {"y": [monthly["Total_Sales"]]},
                            {"yaxis": {"title": "Total Sales"}},
                        ],
                    },
                    {
                        "label": "Quantity Sold",
                        "method": "update",
                        "args": [
                            {"y": [monthly["Quantity"]]},
                            {"yaxis": {"title": "Quantity Sold"}},
                        ],
                    },
                ],
                "direction": "down",
                "showactive": True,
                "x": 1,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
            }
        ]
    )

    # 6. Interactive product bar chart
    product_chart = px.bar(
        products,
        x="Product",
        y="Total_Sales",
        title="Product Performance",
        hover_data={
            "Quantity": True,
            "Total_Sales": ":,.2f",
        },
        template="plotly_white",
    )

    # 7. Interactive regional pie chart
    regional_chart = px.pie(
        regions,
        names="Region",
        values="Total_Sales",
        title="Regional Sales Share",
        hole=0.4,
        template="plotly_white",
    )

    # 8. Customer segmentation chart
    segment_chart = px.bar(
        segments,
        x="Segment",
        y="Revenue",
        title="Customer Segment Revenue",
        hover_data=["Customers"],
        template="plotly_white",
    )

    # 9. Top customers chart
    customer_chart = px.bar(
        customers,
        x="Customer_ID",
        y="Customer_Value",
        title="Top 10 Most Valuable Customers",
        hover_data={
            "Orders": True,
            "Units": True,
            "Customer_Value": ":,.2f",
        },
        template="plotly_white",
    )

    # 10. Animated product sales chart
    animated_data = (
        data.groupby(
            ["Year_Month", "Product"],
            as_index=False,
        )["Total_Sales"]
        .sum()
    )

    animated_chart = px.bar(
        animated_data,
        x="Product",
        y="Total_Sales",
        color="Product",
        animation_frame="Year_Month",
        title="Animated Product Sales by Month",
        template="plotly_white",
    )

    # Build complete HTML dashboard
    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>Interactive Sales Dashboard</title>

        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                background: #f4f6f8;
            }}

            header {{
                background: #16324f;
                color: white;
                padding: 24px;
                text-align: center;
            }}

            .metrics {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                padding: 20px;
            }}

            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }}

            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                padding: 20px;
            }}

            .chart-card {{
                background: white;
                padding: 12px;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            }}

            .full-width {{
                grid-column: 1 / -1;
            }}

            @media (max-width: 900px) {{
                .metrics,
                .dashboard-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>

    <body>
        <header>
            <h1>Interactive Sales Dashboard</h1>
            <p>Sales trends, customer segmentation and product performance</p>
        </header>

        <section class="metrics">
            <div class="metric-card">
                <h3>Total Revenue</h3>
                <p>₹{data["Total_Sales"].sum():,.2f}</p>
            </div>

            <div class="metric-card">
                <h3>Total Quantity</h3>
                <p>{data["Quantity"].sum():,.0f}</p>
            </div>

            <div class="metric-card">
                <h3>Unique Customers</h3>
                <p>{data["Customer_ID"].nunique()}</p>
            </div>

            <div class="metric-card">
                <h3>Top Product</h3>
                <p>{products.iloc[0]["Product"]}</p>
            </div>
        </section>

        <section class="dashboard-grid">
            <div class="chart-card">
                {pio.to_html(
                    monthly_chart,
                    include_plotlyjs="cdn",
                    full_html=False
                )}
            </div>

            <div class="chart-card">
                {pio.to_html(
                    product_chart,
                    include_plotlyjs=False,
                    full_html=False
                )}
            </div>

            <div class="chart-card">
                {pio.to_html(
                    regional_chart,
                    include_plotlyjs=False,
                    full_html=False
                )}
            </div>

            <div class="chart-card">
                {pio.to_html(
                    segment_chart,
                    include_plotlyjs=False,
                    full_html=False
                )}
            </div>

            <div class="chart-card full-width">
                {pio.to_html(
                    customer_chart,
                    include_plotlyjs=False,
                    full_html=False
                )}
            </div>

            <div class="chart-card full-width">
                {pio.to_html(
                    animated_chart,
                    include_plotlyjs=False,
                    full_html=False
                )}
            </div>
        </section>
    </body>
    </html>
    """

    dashboard_file = (
        DASHBOARD_DIR / "interactive_sales_dashboard.html"
    )

    dashboard_file.write_text(
        dashboard_html,
        encoding="utf-8",
    )


# --------------------------------------------------
# Create written report
# --------------------------------------------------
def create_report(
    data: pd.DataFrame,
    analysis: dict,
) -> None:
    """Create a written dashboard report with business insights."""

    products = analysis["product_sales"]
    regions = analysis["regional_sales"]
    customers = analysis["customer_sales"]
    monthly = analysis["monthly_sales"]

    top_product = products.iloc[0]
    top_region = regions.iloc[0]
    top_customer = customers.iloc[0]
    best_month = monthly.loc[
        monthly["Total_Sales"].idxmax()
    ]

    report = f"""# Interactive Sales Dashboard Report

## Executive Summary

This project analyzes sales trends, customer segmentation,
regional performance and product performance using Python,
Pandas, Seaborn, Matplotlib and Plotly.

## Dataset Overview

- Total records: {len(data)}
- Unique products: {data["Product"].nunique()}
- Unique customers: {data["Customer_ID"].nunique()}
- Regions: {data["Region"].nunique()}
- Total revenue: ₹{data["Total_Sales"].sum():,.2f}

## Key Insights

1. **Top product:** {top_product["Product"]} generated
   ₹{top_product["Total_Sales"]:,.2f} in revenue.

2. **Best region:** {top_region["Region"]} generated
   ₹{top_region["Total_Sales"]:,.2f} in revenue.

3. **Most valuable customer:** {top_customer["Customer_ID"]}
   contributed ₹{top_customer["Customer_Value"]:,.2f}.

4. **Best sales month:** {best_month["Year_Month"]} generated
   ₹{best_month["Total_Sales"]:,.2f}.

5. Customer segmentation helps identify high-value customers
   for loyalty offers and personalized promotions.

## Visualizations

- Sales distribution box plot
- Regional violin plot
- Correlation heatmap
- Seaborn 2×2 dashboard
- Interactive monthly sales trend
- Interactive product performance chart
- Regional sales donut chart
- Customer segmentation chart
- Top customer chart
- Animated monthly product chart

## Business Recommendations

- Reward Platinum customers with loyalty programs.
- Increase inventory for top-performing products.
- Run targeted promotions in lower-performing regions.
- Use monthly trends to plan seasonal campaigns.
- Cross-sell related products to high-value customers.

## Conclusion

The dashboard converts raw sales data into meaningful
business insights through statistical and interactive
visualizations.
"""

    report_file = REPORT_DIR / "dashboard_report.md"
    report_file.write_text(report, encoding="utf-8")


# --------------------------------------------------
# Main program
# --------------------------------------------------
def main() -> None:
    """Run the complete sales dashboard pipeline."""

    try:
        print("Loading sales dataset...")
        sales_data = load_data(DATA_FILE)

        print("Cleaning and validating data...")
        sales_data = clean_data(sales_data)

        if sales_data.empty:
            raise ValueError(
                "No valid records remain after data cleaning."
            )

        print("Preparing analysis...")
        analysis = prepare_analysis(sales_data)

        print("Creating Seaborn visualizations...")
        create_seaborn_visualizations(
            sales_data,
            analysis,
        )

        print("Creating interactive Plotly dashboard...")
        create_interactive_dashboard(
            sales_data,
            analysis,
        )

        print("Creating written report...")
        create_report(
            sales_data,
            analysis,
        )

        print("\nDashboard completed successfully.")
        print(
            "Open:",
            DASHBOARD_DIR / "interactive_sales_dashboard.html",
        )

    except FileNotFoundError as error:
        print(f"\nFile error: {error}")

    except ValueError as error:
        print(f"\nValidation error: {error}")

    except pd.errors.EmptyDataError:
        print("\nError: The CSV file is empty.")

    except pd.errors.ParserError:
        print("\nError: The CSV format is invalid.")

    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()