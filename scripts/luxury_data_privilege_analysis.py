from pathlib import Path
import warnings
import random
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Central configuration for the analysis pipeline."""
    
    # Directory paths
    root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = None
    output_dir: Path = None
    interactive_dir: Path = None
    sensitivity_dir: Path = None
    
    # Input files
    main_csv: Path = None
    country_csv: Path = None
    series_csv: Path = None
    multiplier_csv: Path = None
    
    # Analysis parameters
    indicator_cap: int = 300
    use_offset: bool = False
    random_seed: int = 0
    
    # Penalty weights
    fill_penalty_min: float = 0.3
    fill_penalty_max: float = 1.0
    count_penalty_min: float = 0.4
    count_penalty_max: float = 0.6
    
    # TWO MULTIPLIER SETS
    neutral_multipliers: Dict[str, float] = None
    adjusted_multipliers: Dict[str, float] = None
    
    # Keywords for luxury indicator selection
    luxury_keywords: Dict[str, int] = None
    
    def __post_init__(self):
        """Initialize derived paths and default values."""
        if self.data_dir is None:
            self.data_dir = self.root / 'dataset'
        
        if self.output_dir is None:
            self.output_dir = self.root / 'outputs'
            self.output_dir.mkdir(exist_ok=True)
        
        if self.interactive_dir is None:
            self.interactive_dir = self.output_dir / 'interactive'
            self.interactive_dir.mkdir(exist_ok=True)
        
        if self.sensitivity_dir is None:
            self.sensitivity_dir = self.root / 'multiplier_sets'
            self.sensitivity_dir.mkdir(exist_ok=True)
        
        if self.main_csv is None:
            self.main_csv = self.data_dir / 'WDICSV.csv'
        
        if self.country_csv is None:
            self.country_csv = self.data_dir / 'WDICountry.csv'
        
        if self.series_csv is None:
            self.series_csv = self.data_dir / 'WDISeries.csv'
        
        if self.multiplier_csv is None:
            self.multiplier_csv = self.root / 'country_multipliers.csv'
        
        # NEUTRAL multipliers (for objective analysis)
        if self.neutral_multipliers is None:
            self.neutral_multipliers = {
                'High income': 1.0,
                'Upper middle income': 1.0,
                'Lower middle income': 1.0,
                'Low income': 1.0
            }
        
        # ADJUSTED multipliers (original methodology)
        if self.adjusted_multipliers is None:
            self.adjusted_multipliers = {
                'High income': 1.30,
                'Upper middle income': 1.12,
                'Lower middle income': 0.92,
                'Low income': 0.78
            }
        
        if self.luxury_keywords is None:
            self.luxury_keywords = {
                'patent applications': 10,
                'trademark applications': 8,
                'industrial design': 5,
                'r&d expenditure': 8,
                'researchers in r&d': 5,
                'scientific and technical journal': 5,
                'market capitalization': 8,
                'stock market': 8,
                'venture capital': 3,
                'high-technology exports': 8,
                'ict goods exports': 5,
                'fixed broadband subscriptions': 5,
                'individuals using the internet': 5
            }


# ==============================================================================
# DATA LOADING
# ==============================================================================

class DataLoader:
    """Handles loading and validation of WDI datasets."""
    
    def __init__(self, config: Config):
        self.config = config
        self.df_main: Optional[pd.DataFrame] = None
        self.df_country: Optional[pd.DataFrame] = None
        self.df_series: Optional[pd.DataFrame] = None
        self.year_columns: List[str] = []
        self.recent_years: List[str] = []
    
    def validate_files(self) -> None:
        """Check that all required CSV files exist."""
        required_files = [
            self.config.main_csv,
            self.config.country_csv,
            self.config.series_csv
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f'Missing required file: {file_path}')
    
    def load_data(self) -> None:
        """Load all CSV files into DataFrames."""
        print(f'Loading data from: {self.config.data_dir}')

        self.df_main = pd.read_csv(
            self.config.main_csv,
            encoding='utf-8',
            low_memory=False
        )
        
        self.df_country = pd.read_csv(
            self.config.country_csv,
            encoding='utf-8',
            low_memory=False
        )
        
        self.df_series = pd.read_csv(
            self.config.series_csv,
            encoding='utf-8',
            low_memory=False
        )
        
        print(f'Data loaded: Main={self.df_main.shape}, '
                            f'Country={self.df_country.shape}, Series={self.df_series.shape}')

        self._identify_year_columns()
    
    def _identify_year_columns(self) -> None:
        """Extract year columns from the main dataframe."""
        self.year_columns = [
            col for col in self.df_main.columns
            if col.isdigit()
        ]
        
        self.recent_years = [
            str(year) for year in range(2003, 2024)
            if str(year) in self.year_columns
        ]
        
        if self.recent_years:
            print(f'Recent years range: {self.recent_years[0]} - {self.recent_years[-1]}')


# ==============================================================================
# INDICATOR SELECTION
# ==============================================================================

class IndicatorSelector:
    """Selects luxury indicators based on keywords and frequency."""
    
    def __init__(self, config: Config, df_series: pd.DataFrame, df_main: pd.DataFrame):
        self.config = config
        self.df_series = df_series
        self.df_main = df_main
        self.selected_indicators: List[str] = []
    
    def select_indicators(self) -> List[str]:
        """Select indicators based on keywords and frequency."""
        self.selected_indicators = self._select_by_keywords()
        
        if len(self.selected_indicators) < self.config.indicator_cap:
            self._augment_with_frequent()
        print(f'Selected {len(self.selected_indicators)} luxury indicators')
        return self.selected_indicators
    
    def _select_by_keywords(self) -> List[str]:
        """Select indicators matching luxury keywords."""
        indicators = []
        
        for keyword, max_count in self.config.luxury_keywords.items():
            matches = self.df_series[
                self.df_series['Indicator Name'].str.contains(
                    keyword,
                    case=False,
                    na=False,
                    regex=False
                )
            ]['Series Code'].tolist()
            
            indicators.extend(matches[:max_count])
        
        return list(dict.fromkeys(indicators))
    
    def _augment_with_frequent(self) -> None:
        """Add most frequent indicators to reach the cap."""
        frequent = self.df_main['Indicator Code'].value_counts().index.tolist()
        
        for code in frequent:
            if code not in self.selected_indicators:
                self.selected_indicators.append(code)
            
            if len(self.selected_indicators) >= self.config.indicator_cap:
                break


# ==============================================================================
# SCORE COMPUTATION
# ==============================================================================

class ScoreCalculator:
    """Calculates luxury access scores with flexible multiplier support."""
    
    def __init__(
        self,
        config: Config,
        df_main: pd.DataFrame,
        df_country: pd.DataFrame,
        indicators: List[str],
        recent_years: List[str]
    ):
        self.config = config
        self.df_main = df_main
        self.df_country = df_country
        self.indicators = indicators
        self.recent_years = recent_years
        self.country_multipliers: Dict[str, float] = {}
        
        self._load_country_multipliers()
    
    def _load_country_multipliers(self) -> None:
        """Load per-country multiplier overrides from CSV."""
        if not self.config.multiplier_csv.exists():
            return
        
        try:
            df_mult = pd.read_csv(self.config.multiplier_csv)
            
            for _, row in df_mult.iterrows():
                code = str(row.get('Country Code', '')).strip()
                value = row.get('MultiplierPct', row.get('Multiplier', np.nan))
                
                if pd.notna(value):
                    value = float(value)
                    if value > 2:
                        value = value / 100.0
                    self.country_multipliers[code] = value
            
            print(f'📊 Loaded {len(self.country_multipliers)} custom multipliers')
        
        except Exception as e:
            print(f'⚠️  Warning: Could not load multiplier CSV: {e}')
    
    def _get_country_info(self, country_code: str) -> Tuple[str, str, str]:
        """Retrieve country metadata."""
        country_data = self.df_country[
            self.df_country['Country Code'] == country_code
        ]
        
        if country_data.empty:
            return country_code, 'Unknown', 'Unknown'
        
        name = country_data['Short Name'].iloc[0] if 'Short Name' in country_data.columns else country_code
        income = country_data['Income Group'].iloc[0] if 'Income Group' in country_data.columns else 'Unknown'
        region = country_data['Region'].iloc[0] if 'Region' in country_data.columns else 'Unknown'
        
        return name, income, region
    
    def _calculate_penalties(
        self,
        indicator_count: int,
        data_filled_frac: float
    ) -> Tuple[float, float]:
        """Calculate continuous penalties for incomplete data."""
        fill_penalty = np.clip(
            self.config.fill_penalty_min + 
            (self.config.fill_penalty_max - self.config.fill_penalty_min) * data_filled_frac,
            self.config.fill_penalty_min,
            self.config.fill_penalty_max
        )
        
        count_ratio = min(1.0, indicator_count / max(1, self.config.indicator_cap))
        count_penalty = np.clip(
            self.config.count_penalty_min + self.config.count_penalty_max * count_ratio,
            self.config.count_penalty_min,
            1.0
        )
        
        return fill_penalty, count_penalty
    
    def _get_multiplier(
        self,
        country_code: str,
        income_group: str,
        multiplier_set: Dict[str, float]
    ) -> Tuple[float, str]:
        """Determine multiplier for a country."""
        # Priority 1: Country-specific override
        if country_code in self.country_multipliers:
            mult = self.country_multipliers[country_code]
            source = f'custom_csv:{self.config.multiplier_csv.name}'
            return mult, source
        
        # Priority 2: Use provided multiplier set
        mult = multiplier_set.get(income_group, 1.0)
        source = 'income_group_multiplier'
        return mult, source
    
    def compute_scores(
        self,
        multiplier_set: Dict[str, float],
        scoring_type: str = 'neutral'
    ) -> pd.DataFrame:
        """Compute luxury access scores using specified multipliers."""
        
        rows = []
        
        for country_code in self.df_main['Country Code'].unique():
            country_data = self.df_main[
                (self.df_main['Country Code'] == country_code) &
                (self.df_main['Indicator Code'].isin(self.indicators))
            ]
            
            if country_data.empty:
                continue
            
            # Calculate metrics
            indicator_count = len(country_data)
            coverage_frac = indicator_count / max(1, len(self.indicators))
            coverage_pct = coverage_frac * 100.0
            
            total_cells = indicator_count * max(1, len(self.recent_years))
            filled_cells = country_data[self.recent_years].notna().sum().sum() if self.recent_years else 0
            data_filled_frac = filled_cells / total_cells if total_cells > 0 else 0.0
            data_filled_pct = data_filled_frac * 100.0
            
            fill_penalty, count_penalty = self._calculate_penalties(
                indicator_count,
                data_filled_frac
            )
            
            base_score = coverage_pct * fill_penalty * count_penalty
            
            name, income, region = self._get_country_info(country_code)
            multiplier, mult_source = self._get_multiplier(country_code, income, multiplier_set)
            
            adjusted_score = round(base_score * multiplier, 3)
            
            row = {
                'Country Code': country_code,
                'Country Name': name,
                'Income Group': income,
                'Region': region,
                'Base Score': round(base_score, 3),
                'Luxury Access Score': adjusted_score,
                'Multiplier': round(multiplier, 4),
                'Multiplier Source': mult_source,
                'Coverage (%)': round(coverage_pct, 1),
                'Data Filled (%)': round(data_filled_pct, 1),
                'Indicator Count': indicator_count,
                'Scoring Type': scoring_type
            }
            
            rows.append(row)
        
        df_scores = pd.DataFrame(rows)
        
        # Filter aggregates
        df_scores = df_scores[
            ~df_scores['Country Name'].str.contains(
                r'income|World|OECD|Euro(?!pe)|Arab|region|IDA|IBRD|fragile|blend|situations',
                case=False,
                na=False,
                regex=True
            )
        ].copy()
        
        return df_scores


# ==============================================================================
# STATISTICAL ANALYSIS
# ==============================================================================

class StatisticalAnalyzer:
    """Performs statistical tests."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def perform_anova(self, df: pd.DataFrame, output_file: Path, scoring_type: str) -> None:
        """Perform one-way ANOVA."""
        print(f'Performing ANOVA ({scoring_type})...')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'ANOVA: Luxury Access Score by Income Group ({scoring_type.upper()} SCORING)\n')
            f.write('=' * 80 + '\n\n')
            
            if scoring_type == 'neutral':
                f.write('NOTE: Scores are based purely on data availability (multiplier = 1.0).\n')
                f.write('This test shows whether data privilege correlates with income.\n\n')
            else:
                f.write('NOTE: Scores include income-based adjustments.\n')
                f.write('This test shows combined effect of data availability and income.\n\n')
            
            f.write('=' * 80 + '\n\n')
            
            try:
                groups = []
                group_names = []
                
                for name, group in df.groupby('Income Group'):
                    values = group['Luxury Access Score'].dropna().values
                    if len(values) > 1:
                        groups.append(values)
                        group_names.append(name)
                
                if len(groups) < 2:
                    f.write('Insufficient groups\n')
                    print(f'ANOVA skipped ({scoring_type})')
                    return
                
                f_stat, p_value = stats.f_oneway(*groups)
                
                f.write(f'Groups: {", ".join(group_names)}\n')
                f.write(f'Sample sizes: {[len(g) for g in groups]}\n\n')
                f.write(f'F-statistic: {f_stat:.4f}\n')
                f.write(f'p-value: {p_value:.4e}\n\n')
                
                if p_value < 0.001:
                    f.write('Result: HIGHLY SIGNIFICANT (p < 0.001)\n')
                elif p_value < 0.05:
                    f.write('Result: SIGNIFICANT (p < 0.05)\n')
                else:
                    f.write('Result: NOT SIGNIFICANT (p >= 0.05)\n')
                
                print(f'ANOVA ({scoring_type}): F={f_stat:.4f}, p={p_value:.4e}')
            
            except Exception as e:
                f.write(f'Error: {e}\n')
                print(f'ANOVA error ({scoring_type}): {e}')
    
    def perform_posthoc(
        self,
        df: pd.DataFrame,
        txt_file: Path,
        csv_file: Path,
        scoring_type: str
    ) -> None:
        """Perform Tukey HSD post-hoc test."""
        print(f'Performing post-hoc ({scoring_type})...')
        
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            df_clean = df.copy()
            df_clean['Luxury Access Score'] = pd.to_numeric(
                df_clean['Luxury Access Score'],
                errors='coerce'
            )
            df_clean = df_clean.dropna(subset=['Luxury Access Score'])
            
            tukey = pairwise_tukeyhsd(
                endog=df_clean['Luxury Access Score'],
                groups=df_clean['Income Group'],
                alpha=0.05
            )
            
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(f'Tukey HSD Post-Hoc Test ({scoring_type.upper()} SCORING)\n')
                f.write('=' * 80 + '\n\n')
                f.write(str(tukey.summary()))
            
            results_df = pd.DataFrame(
                data=tukey._results_table.data[1:],
                columns=tukey._results_table.data[0]
            )
            results_df.to_csv(csv_file, index=False)
            
            print(f'✅ Tukey HSD ({scoring_type}) saved')
        
        except ImportError:
            print(f'statsmodels not available ({scoring_type})')
            self._fallback_pairwise_tests(df, txt_file, csv_file, scoring_type)
        
        except Exception as e:
            print(f'Tukey HSD failed ({scoring_type}): {e}')
            self._fallback_pairwise_tests(df, txt_file, csv_file, scoring_type)
    
    def _fallback_pairwise_tests(
        self,
        df: pd.DataFrame,
        txt_file: Path,
        csv_file: Path,
        scoring_type: str
    ) -> None:
        """Fallback to pairwise t-tests."""
        from itertools import combinations
        
        groups = {}
        for name, group in df.groupby('Income Group'):
            values = group['Luxury Access Score'].dropna().values
            if len(values) > 1:
                groups[name] = values
        
        results = []
        for (g1, v1), (g2, v2) in combinations(groups.items(), 2):
            t_stat, p_raw = stats.ttest_ind(v1, v2, equal_var=False)
            results.append({
                'group1': g1,
                'group2': g2,
                't-statistic': round(t_stat, 4),
                'p-value': round(p_raw, 6),
                'p-bonferroni': round(min(1.0, p_raw * len(list(combinations(groups.keys(), 2)))), 6)
            })
        
        results_df = pd.DataFrame(results)
        results_df['reject'] = results_df['p-bonferroni'] < 0.05
        results_df.to_csv(csv_file, index=False)
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f'Pairwise t-tests with Bonferroni ({scoring_type.upper()})\n')
            f.write('=' * 80 + '\n\n')
            f.write(results_df.to_string(index=False))
        
        print(f'✅ Pairwise t-tests ({scoring_type}) saved')


# ==============================================================================
# VISUALIZATION
# ==============================================================================

class Visualizer:
    """Creates visualizations with improved color scaling."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_choropleth(
        self,
        df: pd.DataFrame,
        html_file: Path,
        png_file: Optional[Path] = None,
        binned: bool = False,
        scoring_type: str = 'neutral'
    ) -> None:
        """Create choropleth map with optimal color distribution."""
        print(f'🗺️  Creating {"binned " if binned else ""}map ({scoring_type})...')
        
        try:
            if binned:
                df = df.copy()
                df['ScoreBin'] = pd.qcut(
                    df['Luxury Access Score'],
                    q=5,
                    labels=['Very low', 'Low', 'Medium', 'High', 'Very high'],
                    duplicates='drop'
                )
                
                fig = px.choropleth(
                    df,
                    locations='Country Code',
                    locationmode='ISO-3',
                    color='ScoreBin',
                    hover_name='Country Name',
                    hover_data={
                        'Coverage (%)': True,
                        'Data Filled (%)': True,
                        'Indicator Count': True,
                        'Income Group': True
                    },
                    color_discrete_sequence=['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4'],
                    labels={'ScoreBin': 'Data Privilege Level'}
                )
                subtitle = 'Binned by Quintiles'
            else:
                # Use percentile-based range for better distribution
                score_min = df['Luxury Access Score'].quantile(0.05)
                score_max = df['Luxury Access Score'].quantile(0.95)
                
                fig = px.choropleth(
                    df,
                    locations='Country Code',
                    locationmode='ISO-3',
                    color='Luxury Access Score',
                    hover_name='Country Name',
                    hover_data={
                        'Coverage (%)': True,
                        'Data Filled (%)': True,
                        'Indicator Count': True,
                        'Income Group': True,
                        'Luxury Access Score': ':.1f'
                    },
                    color_continuous_scale=[
                        [0.0, '#d73027'],
                        [0.2, '#fc8d59'],
                        [0.4, '#fee08b'],
                        [0.6, '#d9ef8b'],
                        [0.8, '#91cf60'],
                        [1.0, '#1a9850']
                    ],
                    range_color=[score_min, score_max],
                    labels={'Luxury Access Score': 'Score'}
                )
                subtitle = 'Continuous Scale'
            
            scoring_note = 'Neutral scoring (no income adjustments)' if scoring_type == 'neutral' else 'Income-adjusted scoring'
            
            fig.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth'
                ),
                height=700,
                title={
                    'text': f'Luxury Data Privilege Map — {subtitle}<br><sub>{scoring_note}</sub>',
                    'x': 0.5,
                    'xanchor': 'center'
                }
            )
            
            fig.write_html(str(html_file))
            print(f'✅ Map saved: {html_file.name}')
            
            if png_file:
                try:
                    fig.write_image(str(png_file))
                    print(f'PNG saved: {png_file.name}')
                except Exception as e:
                    print(f'PNG export failed: {e}')
        
        except Exception as e:
            print(f'Map creation error: {e}')
    
    def create_tukey_plot(
        self,
        df_cleaned: pd.DataFrame,
        posthoc_csv: Path,
        output_file: Path,
        scoring_type: str
    ) -> None:
        """Create Tukey HSD visualization."""
        print(f'Creating Tukey plot ({scoring_type})...')
        
        try:
            import matplotlib.pyplot as plt
            from math import sqrt
            
            grp_stats = df_cleaned.groupby('Income Group')['Luxury Access Score'].agg([
                'count', 'mean', 'std'
            ]).reset_index()
            
            ci_low, ci_high = [], []
            for _, row in grp_stats.iterrows():
                n = int(row['count'])
                mean = float(row['mean'])
                std = float(row['std']) if pd.notna(row['std']) else 0.0
                
                if n > 1:
                    se = std / sqrt(n)
                    t_crit = stats.t.ppf(0.975, df=n-1)
                    ci_low.append(mean - t_crit * se)
                    ci_high.append(mean + t_crit * se)
                else:
                    ci_low.append(mean)
                    ci_high.append(mean)
            
            grp_stats['ci_low'] = ci_low
            grp_stats['ci_high'] = ci_high
            grp_stats = grp_stats.sort_values('mean', ascending=False)
            
            sig_pairs = self._extract_significant_pairs(posthoc_csv)
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            positions = np.arange(len(grp_stats))
            means = grp_stats['mean'].values
            errs_low = means - grp_stats['ci_low'].values
            errs_high = grp_stats['ci_high'].values - means
            
            colors = ['#2166ac', '#67a9cf', '#fddbc7', '#ef8a62'][:len(grp_stats)]
            
            bars = ax.bar(
                positions,
                means,
                yerr=[errs_low, errs_high],
                capsize=10,
                color=colors,
                alpha=0.85,
                edgecolor='black',
                linewidth=1.5,
                error_kw={'linewidth': 2}
            )
            
            for bar, mean, err_high in zip(bars, means, errs_high):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + err_high + 1,
                    f'{mean:.1f}',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    fontsize=11
                )
            
            ax.set_xticks(positions)
            ax.set_xticklabels(grp_stats['Income Group'], rotation=30, ha='right', fontsize=11)
            ax.set_ylabel('Data Privilege Score (Mean ± 95% CI)', fontsize=13, fontweight='bold')
            
            title_suffix = '(Neutral Scoring)' if scoring_type == 'neutral' else '(Income-Adjusted)'
            ax.set_title(
                f'Data Privilege by Income Group {title_suffix}',
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            self._add_significance_brackets(ax, grp_stats, sig_pairs)
            
            plt.tight_layout()
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f'✅ Tukey plot saved: {output_file.name}')
        
        except Exception as e:
            print(f'Tukey plot error: {e}')
    
    @staticmethod
    def _extract_significant_pairs(csv_file: Path) -> List[Tuple[str, str]]:
        """Extract significant pairs from CSV."""
        sig_pairs = []
        
        if not csv_file.exists():
            return sig_pairs
        
        try:
            df = pd.read_csv(csv_file)
            df.columns = [col.strip() for col in df.columns]
            
            if 'reject' in df.columns:
                for _, row in df.iterrows():
                    if str(row['reject']).lower() in ('true', '1', 'yes'):
                        sig_pairs.append((row.iloc[0], row.iloc[1]))
        except:
            pass
        
        return sig_pairs
    
    @staticmethod
    def _add_significance_brackets(ax, grp_stats, sig_pairs):
        """Add significance brackets."""
        if not sig_pairs:
            return
        
        y_max = grp_stats['ci_high'].max()
        step = y_max * 0.08
        
        group_to_idx = {group: idx for idx, group in enumerate(grp_stats['Income Group'])}
        
        current_y = y_max + step
        for g1, g2 in sig_pairs:
            if g1 not in group_to_idx or g2 not in group_to_idx:
                continue
            
            i = group_to_idx[g1]
            j = group_to_idx[g2]
            
            if i == j:
                continue
            
            if i > j:
                i, j = j, i
            
            y_pos = max(
                grp_stats.iloc[i]['ci_high'],
                grp_stats.iloc[j]['ci_high']
            ) + current_y - y_max
            
            ax.plot([i, i, j, j], [y_pos - step*0.3, y_pos, y_pos, y_pos - step*0.3],
                   lw=1.8, c='black')
            ax.text((i + j) / 2, y_pos + step*0.1, '***', ha='center', va='bottom',
                   fontsize=16, fontweight='bold')
            
            current_y += step


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class LuxuryDataPrivilegePipeline:
    """Main orchestrator with dual scoring system."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self._setup_reproducibility()
        
        self.loader: Optional[DataLoader] = None
        self.selector: Optional[IndicatorSelector] = None
        self.calculator: Optional[ScoreCalculator] = None
        self.analyzer: Optional[StatisticalAnalyzer] = None
        self.visualizer: Optional[Visualizer] = None
    
    def _setup_reproducibility(self) -> None:
        """Set random seeds."""
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
    
    def run(self) -> None:
        """Execute complete dual-scoring pipeline."""
        print('\n' + '=' * 80)
        print('LUXURY DATA PRIVILEGE ANALYSIS - DUAL SCORING SYSTEM')
        print('=' * 80)
        print('Version 1: NEUTRAL (multiplier = 1.0)')
        print('Version 2: ADJUSTED (income-based multipliers)')
        print('=' * 80 + '\n')
        
        # Step 1: Load data
        print('[1/8]Loading data...')
        self.loader = DataLoader(self.config)
        self.loader.validate_files()
        self.loader.load_data()
        print()
        
        # Step 2: Select indicators
        print('[2/8]Selecting indicators...')
        self.selector = IndicatorSelector(
            self.config,
            self.loader.df_series,
            self.loader.df_main
        )
        indicators = self.selector.select_indicators()
        print()
        
        # Step 3: Initialize calculator
        print('[3/8]Initializing score calculator...')
        self.calculator = ScoreCalculator(
            self.config,
            self.loader.df_main,
            self.loader.df_country,
            indicators,
            self.loader.recent_years
        )
        print()
        
        # Step 4: Compute NEUTRAL scores
        print('[4/8]Computing NEUTRAL scores...')
        df_neutral = self.calculator.compute_scores(
            multiplier_set=self.config.neutral_multipliers,
            scoring_type='neutral'
        )
        neutral_file = self.config.output_dir / 'luxury_scores_neutral.csv'
        df_neutral.to_csv(neutral_file, index=False)
        print(f'Saved: {neutral_file.name}')
        
        self._show_top10(df_neutral, 'NEUTRAL')
        print()
        
        # Step 5: Compute ADJUSTED scores
        print('[5/8]Computing ADJUSTED scores...')
        df_adjusted = self.calculator.compute_scores(
            multiplier_set=self.config.adjusted_multipliers,
            scoring_type='adjusted'
        )
        adjusted_file = self.config.output_dir / 'luxury_scores_adjusted.csv'
        df_adjusted.to_csv(adjusted_file, index=False)
        print(f'Saved: {adjusted_file.name}')
        
        self._show_top10(df_adjusted, 'ADJUSTED')
        print()
        
        # Step 6: Statistical analysis
        print('[6/8]Running statistical tests...')
        self.analyzer = StatisticalAnalyzer(self.config)
        
        # Clean data
        df_neutral_clean = self._clean_for_stats(df_neutral)
        df_adjusted_clean = self._clean_for_stats(df_adjusted)
        
        df_neutral_clean.to_csv(
            self.config.output_dir / 'luxury_scores_neutral_cleaned.csv',
            index=False
        )
        df_adjusted_clean.to_csv(
            self.config.output_dir / 'luxury_scores_adjusted_cleaned.csv',
            index=False
        )
        
        # ANOVA - Neutral
        self.analyzer.perform_anova(
            df_neutral_clean,
            self.config.output_dir / 'anova_neutral.txt',
            'neutral'
        )
        
        # ANOVA - Adjusted
        self.analyzer.perform_anova(
            df_adjusted_clean,
            self.config.output_dir / 'anova_adjusted.txt',
            'adjusted'
        )
        
        # Post-hoc - Neutral
        self.analyzer.perform_posthoc(
            df_neutral_clean,
            self.config.output_dir / 'posthoc_neutral.txt',
            self.config.output_dir / 'posthoc_neutral.csv',
            'neutral'
        )
        
        # Post-hoc - Adjusted
        self.analyzer.perform_posthoc(
            df_adjusted_clean,
            self.config.output_dir / 'posthoc_adjusted.txt',
            self.config.output_dir / 'posthoc_adjusted.csv',
            'adjusted'
        )
        print()
        
        # Step 7: Create visualizations
        print('[7/8]Creating visualizations...')
        self.visualizer = Visualizer(self.config)
        
        # Neutral maps
        self.visualizer.create_choropleth(
            df_neutral,
            self.config.interactive_dir / 'map_neutral.html',
            self.config.output_dir / 'map_neutral.png',
            binned=False,
            scoring_type='neutral'
        )
        
        self.visualizer.create_choropleth(
            df_neutral,
            self.config.interactive_dir / 'map_neutral_binned.html',
            self.config.output_dir / 'map_neutral_binned.png',
            binned=True,
            scoring_type='neutral'
        )
        
        # Adjusted maps
        self.visualizer.create_choropleth(
            df_adjusted,
            self.config.interactive_dir / 'map_adjusted.html',
            self.config.output_dir / 'map_adjusted.png',
            binned=False,
            scoring_type='adjusted'
        )
        
        self.visualizer.create_choropleth(
            df_adjusted,
            self.config.interactive_dir / 'map_adjusted_binned.html',
            self.config.output_dir / 'map_adjusted_binned.png',
            binned=True,
            scoring_type='adjusted'
        )
        
        # Tukey plots
        self.visualizer.create_tukey_plot(
            df_neutral_clean,
            self.config.output_dir / 'posthoc_neutral.csv',
            self.config.output_dir / 'tukey_plot_neutral.png',
            'neutral'
        )
        
        self.visualizer.create_tukey_plot(
            df_adjusted_clean,
            self.config.output_dir / 'posthoc_adjusted.csv',
            self.config.output_dir / 'tukey_plot_adjusted.png',
            'adjusted'
        )
        print()
        
        # Step 8: Write metadata
        print('[8/8] Writing metadata...')
        self._write_metadata()
        print()
        
        print('=' * 80)
        print(' PIPELINE COMPLETED SUCCESSFULLY')
        print('=' * 80)
        print()
        print(f' Output directory: {self.config.output_dir}')
        print(f'  Interactive maps: {self.config.interactive_dir}')
        print()
        print(' Key outputs:')
        print('   NEUTRAL SCORING:')
        print('     • luxury_scores_neutral.csv')
        print('     • anova_neutral.txt')
        print('     • posthoc_neutral.txt / .csv')
        print('     • tukey_plot_neutral.png')
        print('     • map_neutral.html / .png')
        print()
        print('   ADJUSTED SCORING:')
        print('     • luxury_scores_adjusted.csv')
        print('     • anova_adjusted.txt')
        print('     • posthoc_adjusted.txt / .csv')
        print('     • tukey_plot_adjusted.png')
        print('     • map_adjusted.html / .png')
        print()
    
    @staticmethod
    def _clean_for_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Clean data for statistical tests."""
        return df[
            df['Income Group'].notna() &
            (df['Income Group'].str.strip().str.lower() != 'unknown')
        ].copy()
    
    @staticmethod
    def _show_top10(df: pd.DataFrame, scoring_type: str) -> None:
        """Display top 10 countries."""
        print(f'\n Top 10 Countries ({scoring_type}):')
        top10 = df.nlargest(10, 'Luxury Access Score')[
            ['Country Name', 'Luxury Access Score', 'Income Group', 'Coverage (%)']
        ]
        for i, row in enumerate(top10.itertuples(), 1):
            print(f'  {i:2d}. {row._1:30s} {row._2:6.1f}  ({row._3})')
    
    def _write_metadata(self) -> None:
        """Write comprehensive metadata."""
        readme_file = self.config.output_dir / 'README.txt'
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write('LUXURY DATA PRIVILEGE ANALYSIS - DUAL SCORING SYSTEM\n')
            f.write('=' * 80 + '\n\n')
            
            f.write('OVERVIEW\n')
            f.write('-' * 80 + '\n')
            f.write('This analysis provides TWO scoring approaches:\n\n')
            
            f.write('1. NEUTRAL SCORING (Recommended for Academic Use)\n')
            f.write('   - Multiplier = 1.0 for ALL countries\n')
            f.write('   - Scores based purely on data availability\n')
            f.write('   - Objective measurement of data privilege\n')
            f.write('   - Shows natural correlation between income and data quality\n\n')
            
            f.write('2. ADJUSTED SCORING (Comparative Analysis)\n')
            f.write('   - Income-based multipliers applied:\n')
            for group, mult in self.config.adjusted_multipliers.items():
                f.write(f'     • {group:25s}: {mult:.2f}\n')
            f.write('   - Amplifies differences between income groups\n')
            f.write('   - Useful for sensitivity analysis\n\n')
            
            f.write('SCORING METHODOLOGY\n')
            f.write('-' * 80 + '\n')
            f.write('Base Score = Coverage × Fill_Penalty × Count_Penalty\n')
            f.write('Final Score = Base Score × Multiplier\n\n')
            f.write('Where:\n')
            f.write('  • Coverage: % of luxury indicators available\n')
            f.write('  • Fill_Penalty: Based on data completeness (2003-2023)\n')
            f.write('  • Count_Penalty: Based on absolute indicator count\n')
            f.write('  • Multiplier: 1.0 (neutral) or income-based (adjusted)\n\n')
            
            f.write('CONFIGURATION\n')
            f.write('-' * 80 + '\n')
            f.write(f'Indicator cap: {self.config.indicator_cap}\n')
            f.write(f'Random seed: {self.config.random_seed}\n')
            f.write(f'Use offset: {self.config.use_offset}\n\n')
            
            f.write('OUTPUT FILES\n')
            f.write('-' * 80 + '\n')
            f.write('NEUTRAL:\n')
            f.write('  luxury_scores_neutral.csv - All country scores\n')
            f.write('  anova_neutral.txt - ANOVA results\n')
            f.write('  posthoc_neutral.txt/.csv - Post-hoc comparisons\n')
            f.write('  tukey_plot_neutral.png - Visual comparison\n')
            f.write('  interactive/map_neutral.html - Interactive map\n')
            f.write('  interactive/map_neutral_binned.html - Binned map\n\n')
            f.write('ADJUSTED:\n')
            f.write('  luxury_scores_adjusted.csv - All country scores\n')
            f.write('  anova_adjusted.txt - ANOVA results\n')
            f.write('  posthoc_adjusted.txt/.csv - Post-hoc comparisons\n')
            f.write('  tukey_plot_adjusted.png - Visual comparison\n')
            f.write('  interactive/map_adjusted.html - Interactive map\n')
            f.write('  interactive/map_adjusted_binned.html - Binned map\n\n')
            
            f.write('INTERPRETATION\n')
            f.write('-' * 80 + '\n')
            f.write('NEUTRAL SCORES show the objective reality:\n')
            f.write('  • High-income countries have better data infrastructure\n')
            f.write('  • Data privilege correlates strongly with economic development\n')
            f.write('  • No artificial adjustments applied\n\n')
            f.write('ADJUSTED SCORES show comparative perspective:\n')
            f.write('  • Amplifies differences for policy discussions\n')
            f.write('  • Helps visualize the compound effect of income + data quality\n')
            f.write('  • Useful for highlighting disparities\n\n')
            
            f.write('RECOMMENDED USAGE\n')
            f.write('-' * 80 + '\n')
            f.write('For academic papers: Use NEUTRAL scoring\n')
            f.write('For policy advocacy: Compare NEUTRAL vs ADJUSTED\n')
            f.write('For presentations: Use NEUTRAL maps with clear methodology notes\n\n')
            
            f.write('ETHICAL CONSIDERATIONS\n')
            f.write('-' * 80 + '\n')
            f.write('This analysis reveals structural inequalities in global data systems.\n')
            f.write('By providing both neutral and adjusted scores, we offer:\n')
            f.write('  • Objective measurement (neutral)\n')
            f.write('  • Contextual understanding (adjusted)\n')
            f.write('  • Transparency in methodology\n\n')
            
            f.write('DATA SOURCE\n')
            f.write('-' * 80 + '\n')
            f.write('World Bank World Development Indicators (WDI)\n')
            f.write('https://datacatalog.worldbank.org/dataset/world-development-indicators\n\n')
            
            f.write('ANALYSIS DATE\n')
            f.write('-' * 80 + '\n')
            f.write(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        
        print(f'Metadata saved: {readme_file.name}')


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Main entry point."""
    try:
        pipeline = LuxuryDataPrivilegePipeline()
        pipeline.run()
        return 0
        
    except KeyboardInterrupt:
        print('\n\n  Analysis interrupted by user')
        return 130
        
    except Exception as e:
        print(f'\n ERROR: {e}')
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())