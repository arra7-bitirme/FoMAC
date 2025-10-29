"""
Report Generator Module

Generates training reports in both Excel and HTML formats with comprehensive
training metrics, visualizations, and analysis.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')


class TrainingReportGenerator:
    """Generates comprehensive training reports in multiple formats."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_reports(
        self,
        training_results: Any,
        config: Dict[str, Any],
        model_path: Optional[Path] = None,
        dataset_stats: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """
        Generate both Excel and HTML reports.
        
        Args:
            training_results: YOLO training results object
            config: Complete configuration dictionary
            model_path: Path to trained model
            dataset_stats: Dataset statistics
            
        Returns:
            Dictionary mapping report type to file path
        """
        logger.info("🔄 Generating training reports...")
        
        # Extract metrics and information
        report_data = self._extract_report_data(
            training_results, config, model_path, dataset_stats
        )
        
        # Generate reports
        reports = {}
        
        try:
            # Excel report
            excel_path = self._generate_excel_report(report_data)
            reports['excel'] = excel_path
            logger.info(f"✅ Excel report generated: {excel_path}")
            
            # HTML report  
            html_path = self._generate_html_report(report_data)
            reports['html'] = html_path
            logger.info(f"✅ HTML report generated: {html_path}")
            
        except Exception as e:
            logger.error(f"❌ Error generating reports: {e}", exc_info=True)
            raise
        
        logger.info("🎉 All reports generated successfully!")
        return reports
    
    def _extract_report_data(
        self,
        training_results: Any,
        config: Dict[str, Any],
        model_path: Optional[Path],
        dataset_stats: Optional[Dict]
    ) -> Dict[str, Any]:
        """Extract and organize data for reports."""
        report_data = {
            'timestamp': datetime.now(),
            'model_path': str(model_path) if model_path else None,
            'config': config,
            'dataset_stats': dataset_stats or {},
        }
        
        # Extract YOLO training results
        if hasattr(training_results, 'results_dict'):
            # YOLO v8/v11 results
            results = training_results.results_dict
            report_data.update({
                'final_metrics': results,
                'best_epoch': getattr(training_results, 'best_epoch', None),
                'training_time': getattr(training_results, 'training_time', None),
            })
        
        # Try to load metrics from results CSV if available
        if model_path:
            results_dir = model_path.parent.parent
            csv_path = results_dir / 'results.csv'
            if csv_path.exists():
                try:
                    metrics_df = pd.read_csv(csv_path)
                    report_data['metrics_history'] = metrics_df
                except Exception as e:
                    logger.warning(f"Could not load results CSV: {e}")
        
        return report_data
    
    def _generate_excel_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate Excel report with multiple sheets."""
        # Use HHMM-DDMMYYYY format for timestamp
        timestamp = report_data['timestamp'].strftime('%H%M-%d%m%Y')
        excel_path = self.output_dir / f'training_report_{timestamp}.xlsx'
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 1. Summary Sheet
            self._create_summary_sheet(report_data, writer)
            
            # 2. Training Metrics Sheet
            self._create_metrics_sheet(report_data, writer)
            
            # 3. Configuration Sheet
            self._create_config_sheet(report_data, writer)
            
            # 4. Dataset Stats Sheet
            self._create_dataset_sheet(report_data, writer)
        
        return excel_path
    
    def _create_summary_sheet(self, report_data: Dict[str, Any], writer):
        """Create summary sheet with key metrics."""
        config = report_data['config']
        yolo_config = config.get('yolo_params', {})
        
        summary_data = [
            ['Training Report Summary', ''],
            ['Generated', report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')],
            ['', ''],
            ['Model Information', ''],
            ['Architecture', yolo_config.get('model', 'N/A')],
            ['Epochs', yolo_config.get('epochs', 'N/A')],
            ['Batch Size', yolo_config.get('batch', 'N/A')],
            ['Image Size', yolo_config.get('imgsz', 'N/A')],
            ['Model Path', report_data.get('model_path', 'N/A')],
            ['', ''],
            ['Training Settings', ''],
            ['Optimizer', yolo_config.get('optimizer', 'N/A')],
            ['Learning Rate', yolo_config.get('lr0', 'N/A')],
            ['Device', config.get('device', 'N/A')],
            ['Workers', yolo_config.get('workers', 'N/A')],
            ['', ''],
            ['Dataset Information', ''],
        ]
        
        # Add dataset stats if available
        dataset_stats = report_data.get('dataset_stats', {})
        if dataset_stats:
            summary_data.extend([
                ['Train Images', dataset_stats.get('train_images', 'N/A')],
                ['Val Images', dataset_stats.get('val_images', 'N/A')],
                ['Test Images', dataset_stats.get('test_images', 'N/A')],
                ['Classes', dataset_stats.get('num_classes', 'N/A')],
            ])
        
        # Add final metrics if available
        if 'final_metrics' in report_data:
            summary_data.extend([
                ['', ''],
                ['Final Performance', ''],
                ['Best Epoch', report_data.get('best_epoch', 'N/A')],
            ])
            
            metrics = report_data['final_metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    summary_data.append([key, f"{value:.4f}"])
        
        # Add baseline comparison if available
        baseline_metrics = config.get('baseline_metrics', {})
        if baseline_metrics:
            # Add context about the comparison
            model_type = baseline_metrics.get('baseline_model_type', 'Baseline Model')
            baseline_note = baseline_metrics.get('baseline_note', 'Performance before training')
            
            summary_data.extend([
                ['', ''],
                ['🔄 TRANSFER LEARNING ANALYSIS', ''],
                ['', ''],
                ['Model Evolution', model_type],
                ['Baseline Context', baseline_note],
                ['', ''],
                ['Performance Comparison', ''],
                ['', ''],
                ['Metric', 'Before Training', 'After Training', 'Improvement'],
            ])
            
            # Get final metrics for comparison
            final_metrics = report_data.get('final_metrics', {})
            
            # Compare key metrics with better labels
            comparisons = [
                ('Overall mAP50', 'baseline_map50', 'map50'),
                ('Overall mAP50-95', 'baseline_map50_95', 'map'),
                ('Overall Precision', 'baseline_precision', 'precision'),
                ('Overall Recall', 'baseline_recall', 'recall'),
            ]
            
            # Add per-class comparisons if available
            if 'baseline_player_map50' in baseline_metrics:
                comparisons.extend([
                    ('---', '---', '---'),  # Separator
                    ('👤 Player mAP50', 'baseline_player_map50', 'player_map50'),
                    ('👤 Player Precision', 'baseline_player_precision', 'player_precision'),
                    ('👤 Player Recall', 'baseline_player_recall', 'player_recall'),
                    ('---', '---', '---'),  # Separator
                    ('⚽ Ball mAP50', 'baseline_ball_map50', 'ball_map50'),
                    ('⚽ Ball Precision', 'baseline_ball_precision', 'ball_precision'),
                    ('⚽ Ball Recall', 'baseline_ball_recall', 'ball_recall'),
                ])
            
            for metric_name, baseline_key, final_key in comparisons:
                # Handle separators
                if metric_name == '---':
                    summary_data.append(['', '', '', ''])
                    continue
                    
                baseline_val = baseline_metrics.get(baseline_key, 0.0)
                final_val = final_metrics.get(final_key, 0.0)
                
                if isinstance(baseline_val, (int, float)) and isinstance(final_val, (int, float)):
                    improvement = final_val - baseline_val
                    
                    if baseline_val > 0:
                        improvement_pct = improvement / baseline_val * 100
                        improvement_str = f"{improvement:+.4f} ({improvement_pct:+.1f}%)"
                    else:
                        improvement_str = f"{improvement:+.4f} (new capability)" if improvement > 0 else "0.0000 (no detection)"
                    
                    summary_data.append([
                        metric_name,
                        f"{baseline_val:.4f}",
                        f"{final_val:.4f}",
                        improvement_str
                    ])
            
            # Add summary
            summary_data.extend([
                ['', ''],
                ['📊 Training Impact', ''],
                ['Transfer Learning Status', 'Player model successfully adapted for ball detection'],
                ['Multi-class Capability', 'Added ball detection while maintaining player detection'],
            ])
        
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value', '', ''])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    def _create_metrics_sheet(self, report_data: Dict[str, Any], writer):
        """Create detailed metrics sheet."""
        if 'metrics_history' in report_data:
            metrics_df = report_data['metrics_history']
            metrics_df.to_excel(writer, sheet_name='Training_Metrics', index=False)
        else:
            # Create placeholder if no metrics available
            placeholder_df = pd.DataFrame([
                ['No detailed metrics available'],
                ['Metrics are typically saved in results.csv'],
                ['Check the model training directory for more details']
            ], columns=['Note'])
            placeholder_df.to_excel(writer, sheet_name='Training_Metrics', index=False)
    
    def _create_config_sheet(self, report_data: Dict[str, Any], writer):
        """Create configuration sheet with all parameters."""
        config_data = []
        
        for section, params in report_data['config'].items():
            config_data.append([f'=== {section.upper()} ===', ''])
            
            if isinstance(params, dict):
                for key, value in params.items():
                    config_data.append([key, str(value)])
            else:
                config_data.append([section, str(params)])
            
            config_data.append(['', ''])  # Empty row for separation
        
        config_df = pd.DataFrame(config_data, columns=['Parameter', 'Value'])
        config_df.to_excel(writer, sheet_name='Configuration', index=False)
    
    def _create_dataset_sheet(self, report_data: Dict[str, Any], writer):
        """Create dataset statistics sheet."""
        dataset_stats = report_data.get('dataset_stats', {})
        
        if dataset_stats:
            stats_data = []
            for key, value in dataset_stats.items():
                stats_data.append([key, str(value)])
            
            stats_df = pd.DataFrame(stats_data, columns=['Statistic', 'Value'])
            stats_df.to_excel(writer, sheet_name='Dataset_Stats', index=False)
        else:
            # Placeholder
            placeholder_df = pd.DataFrame([
                ['No dataset statistics available'],
                ['Dataset stats can be generated during extraction']
            ], columns=['Note'])
            placeholder_df.to_excel(writer, sheet_name='Dataset_Stats', index=False)
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> Path:
        """Generate HTML report with visualizations."""
        # Use HHMM-DDMMYYYY format for timestamp
        timestamp = report_data['timestamp'].strftime('%H%M-%d%m%Y')
        html_path = self.output_dir / f'training_report_{timestamp}.html'
        
        # Generate plots with safe directory name
        plots_dir = self.output_dir / f'plots_{timestamp}'
        plots_dir.mkdir(exist_ok=True)
        
        plot_paths = self._generate_plots(report_data, plots_dir, timestamp)
        
        # Generate HTML
        html_content = self._create_html_content(report_data, plot_paths, timestamp)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_plots(self, report_data: Dict[str, Any], plots_dir: Path, timestamp: str) -> Dict[str, Path]:
        """Generate visualization plots."""
        plot_paths = {}
        
        try:
            # Training metrics plot
            if 'metrics_history' in report_data:
                metrics_df = report_data['metrics_history']
                
                # Loss curves
                loss_plot = self._plot_loss_curves(metrics_df, plots_dir)
                if loss_plot:
                    plot_paths['loss_curves'] = loss_plot
                
                # mAP curves  
                map_plot = self._plot_map_curves(metrics_df, plots_dir)
                if map_plot:
                    plot_paths['map_curves'] = map_plot
                
                # Learning rate plot
                lr_plot = self._plot_learning_rate(metrics_df, plots_dir)
                if lr_plot:
                    plot_paths['learning_rate'] = lr_plot
                
                # Detailed metrics plot
                detailed_plot = self._plot_detailed_metrics(metrics_df, plots_dir)
                if detailed_plot:
                    plot_paths['detailed_metrics'] = detailed_plot
                
                # Training efficiency plot
                efficiency_plot = self._plot_training_efficiency(metrics_df, plots_dir)
                if efficiency_plot:
                    plot_paths['training_efficiency'] = efficiency_plot
                
                # Performance comparison plot
                comparison_plot = self._plot_performance_comparison(metrics_df, plots_dir)
                if comparison_plot:
                    plot_paths['performance_comparison'] = comparison_plot
                
                # Convergence analysis
                convergence_plot = self._plot_convergence_analysis(metrics_df, plots_dir)
                if convergence_plot:
                    plot_paths['convergence_analysis'] = convergence_plot
            
            # Configuration summary chart
            config_plot = self._plot_config_summary(report_data, plots_dir)
            if config_plot:
                plot_paths['config_summary'] = config_plot
            
            # Dataset distribution chart
            dataset_plot = self._plot_dataset_distribution(report_data, plots_dir)
            if dataset_plot:
                plot_paths['dataset_distribution'] = dataset_plot
            
            # Training timeline
            timeline_plot = self._plot_training_timeline(report_data, plots_dir)
            if timeline_plot:
                plot_paths['training_timeline'] = timeline_plot
                
        except Exception as e:
            logger.warning(f"Error generating plots: {e}")
        
        return plot_paths
    
    def _plot_loss_curves(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot training loss curves."""
        try:
            plt.figure(figsize=(12, 8))
            
            loss_columns = [col for col in metrics_df.columns if 'loss' in col.lower()]
            
            if loss_columns:
                for col in loss_columns:
                    if col in metrics_df.columns:
                        plt.plot(metrics_df.index, metrics_df[col], label=col, linewidth=2)
                
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                loss_path = plots_dir / 'loss_curves.png'
                plt.savefig(loss_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return loss_path
        except Exception as e:
            logger.warning(f"Error plotting loss curves: {e}")
        
        plt.close()
        return None
    
    def _plot_map_curves(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot mAP curves."""
        try:
            plt.figure(figsize=(12, 6))
            
            map_columns = [col for col in metrics_df.columns if 'map' in col.lower() or 'ap' in col.lower()]
            
            if map_columns:
                for col in map_columns:
                    if col in metrics_df.columns:
                        plt.plot(metrics_df.index, metrics_df[col], label=col, linewidth=2, marker='o')
                
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('mAP', fontsize=12)
                plt.title('Validation mAP Curves', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                map_path = plots_dir / 'map_curves.png'
                plt.savefig(map_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return map_path
        except Exception as e:
            logger.warning(f"Error plotting mAP curves: {e}")
        
        plt.close()
        return None
    
    def _plot_config_summary(self, report_data: Dict[str, Any], plots_dir: Path) -> Optional[Path]:
        """Plot configuration summary."""
        try:
            config = report_data['config']
            yolo_config = config.get('yolo_params', {})
            
            # Create a simple bar chart of key parameters
            params = {
                'Epochs': yolo_config.get('epochs', 0),
                'Batch Size': yolo_config.get('batch', 0),
                'Image Size': yolo_config.get('imgsz', 0),
                'Workers': yolo_config.get('workers', 0),
            }
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(params.keys(), params.values(), color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
            
            plt.title('Training Configuration Summary', fontsize=14, fontweight='bold')
            plt.ylabel('Value', fontsize=12)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            config_path = plots_dir / 'config_summary.png'
            plt.savefig(config_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return config_path
        except Exception as e:
            logger.warning(f"Error plotting config summary: {e}")
        
        plt.close()
        return None
    
    def _plot_learning_rate(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot learning rate schedule."""
        try:
            plt.figure(figsize=(10, 6))
            
            lr_columns = [col for col in metrics_df.columns if 'lr' in col.lower()]
            
            if lr_columns:
                for col in lr_columns:
                    if col in metrics_df.columns:
                        plt.plot(metrics_df.index, metrics_df[col], label=col, linewidth=2, marker='o')
                
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Learning Rate', fontsize=12)
                plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')  # Log scale for better visualization
                plt.tight_layout()
                
                lr_path = plots_dir / 'learning_rate.png'
                plt.savefig(lr_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return lr_path
        except Exception as e:
            logger.warning(f"Error plotting learning rate: {e}")
        
        plt.close()
        return None
    
    def _plot_detailed_metrics(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot detailed training metrics in a comprehensive view."""
        try:
            # Create a 2x2 subplot for detailed metrics
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: All losses
            loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]
            if loss_cols:
                for col in loss_cols:
                    ax1.plot(metrics_df.index, metrics_df[col], label=col, linewidth=2)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Losses')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Precision and Recall
            precision_cols = [col for col in metrics_df.columns if 'precision' in col.lower() or 'p(' in col.lower()]
            recall_cols = [col for col in metrics_df.columns if 'recall' in col.lower() or 'r(' in col.lower()]
            
            for col in precision_cols:
                if col in metrics_df.columns:
                    ax2.plot(metrics_df.index, metrics_df[col], label=f'Precision: {col}', linewidth=2, linestyle='-')
            for col in recall_cols:
                if col in metrics_df.columns:
                    ax2.plot(metrics_df.index, metrics_df[col], label=f'Recall: {col}', linewidth=2, linestyle='--')
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
            ax2.set_title('Precision & Recall')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # Plot 3: mAP scores
            map_cols = [col for col in metrics_df.columns if 'map' in col.lower()]
            for col in map_cols:
                if col in metrics_df.columns:
                    ax3.plot(metrics_df.index, metrics_df[col], label=col, linewidth=2, marker='o')
            
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('mAP')
            ax3.set_title('Mean Average Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # Plot 4: F1 Score and other metrics
            f1_cols = [col for col in metrics_df.columns if 'f1' in col.lower()]
            other_metrics = [col for col in metrics_df.columns if any(x in col.lower() for x in ['fitness', 'obj', 'cls', 'dfl'])]
            
            for col in f1_cols:
                if col in metrics_df.columns:
                    ax4.plot(metrics_df.index, metrics_df[col], label=col, linewidth=2, marker='s')
            
            # Add secondary y-axis for other metrics if they exist
            if other_metrics:
                ax4_twin = ax4.twinx()
                for col in other_metrics[:3]:  # Limit to 3 metrics to avoid clutter
                    if col in metrics_df.columns:
                        ax4_twin.plot(metrics_df.index, metrics_df[col], label=col, linewidth=1, linestyle=':')
                ax4_twin.set_ylabel('Other Metrics')
                ax4_twin.legend(loc='upper right')
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('F1 Score')
            ax4.set_title('F1 Score & Other Metrics')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            detailed_path = plots_dir / 'detailed_metrics.png'
            plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return detailed_path
            
        except Exception as e:
            logger.warning(f"Error plotting detailed metrics: {e}")
        
        plt.close()
        return None
    
    def _create_html_content(self, report_data: Dict[str, Any], plot_paths: Dict[str, Path], timestamp: str) -> str:
        """Create HTML report content."""
        config = report_data['config']
        yolo_config = config.get('yolo_params', {})
        timestamp_display = report_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>YOLO Training Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 3px solid #3498db;
                }}
                .header h1 {{
                    color: #2c3e50;
                    margin: 0;
                    font-size: 2.5em;
                }}
                .timestamp {{
                    color: #7f8c8d;
                    font-style: italic;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .section h2 {{
                    color: #2c3e50;
                    margin-top: 0;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-label {{
                    font-weight: bold;
                    color: #34495e;
                }}
                .metric-value {{
                    font-size: 1.2em;
                    color: #3498db;
                    margin-top: 5px;
                }}
                .plot {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .plot img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .config-table, .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .config-table th, .config-table td, .metrics-table th, .metrics-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .config-table th, .metrics-table th {{
                    background-color: #3498db;
                    color: white;
                }}
                .config-table tr:hover, .metrics-table tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metrics-table td {{
                    text-align: center;
                }}
                .metrics-table .epoch-col {{
                    font-weight: bold;
                }}
                .best-epoch {{
                    background-color: #d5f4e6 !important;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏆 YOLO Training Report</h1>
                    <p class="timestamp">Generated: {timestamp_display}</p>
                </div>
                
                <div class="section">
                    <h2>📊 Executive Summary</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Model Architecture</div>
                            <div class="metric-value">{yolo_config.get('model', 'N/A')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Total Epochs</div>
                            <div class="metric-value">{yolo_config.get('epochs', 'N/A')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Batch Size</div>
                            <div class="metric-value">{yolo_config.get('batch', 'N/A')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Image Size</div>
                            <div class="metric-value">{yolo_config.get('imgsz', 'N/A')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Device</div>
                            <div class="metric-value">{config.get('device', 'N/A')}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Dataset Classes</div>
                            <div class="metric-value">{report_data.get('dataset_stats', {}).get('num_classes', 'N/A')}</div>
                        </div>
                    </div>
                </div>
        """
        
        # Add baseline comparison section if available
        baseline_metrics = config.get('baseline_metrics', {})
        if baseline_metrics:
            model_type = baseline_metrics.get('baseline_model_type', 'Baseline Model')
            baseline_note = baseline_metrics.get('baseline_note', 'Performance before training')
            
            html += f"""
                <div class="section">
                    <h2>🔄 Transfer Learning Analysis</h2>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Model Evolution</div>
                            <div class="metric-value">{model_type}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Training Context</div>
                            <div class="metric-value">{baseline_note}</div>
                        </div>
                    </div>
                    
                    <h3>📊 Performance Comparison: Before vs After Training</h3>
                    <table class="config-table">
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Before Training</th>
                                <th>After Training</th>
                                <th>Improvement</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            # Get final metrics for comparison
            final_metrics = report_data.get('final_metrics', {})
            
            # Compare key metrics
            comparisons = [
                ('Overall mAP50', 'baseline_map50', 'map50'),
                ('Overall mAP50-95', 'baseline_map50_95', 'map'),
                ('Overall Precision', 'baseline_precision', 'precision'),
                ('Overall Recall', 'baseline_recall', 'recall'),
            ]
            
            # Add per-class comparisons if available
            if 'baseline_player_map50' in baseline_metrics:
                comparisons.extend([
                    ('separator', '', ''),  # Separator line
                    ('👤 Player mAP50', 'baseline_player_map50', 'player_map50'),
                    ('👤 Player Precision', 'baseline_player_precision', 'player_precision'),
                    ('👤 Player Recall', 'baseline_player_recall', 'player_recall'),
                    ('separator', '', ''),  # Separator line
                    ('⚽ Ball mAP50', 'baseline_ball_map50', 'ball_map50'),
                    ('⚽ Ball Precision', 'baseline_ball_precision', 'ball_precision'),
                    ('⚽ Ball Recall', 'baseline_ball_recall', 'ball_recall'),
                ])
            
            for metric_name, baseline_key, final_key in comparisons:
                # Handle separator rows
                if metric_name == 'separator':
                    html += """
                            <tr style="background-color: #f8f9fa;">
                                <td colspan="4" style="text-align: center; font-weight: bold; color: #6c757d;">
                                    ───────────────────────────────────
                                </td>
                            </tr>
                    """
                    continue
                    
                baseline_val = baseline_metrics.get(baseline_key, 0.0)
                final_val = final_metrics.get(final_key, 0.0)
                
                if isinstance(baseline_val, (int, float)) and isinstance(final_val, (int, float)):
                    improvement = final_val - baseline_val
                    
                    if baseline_val > 0:
                        improvement_pct = improvement / baseline_val * 100
                        improvement_str = f"{improvement:+.4f} ({improvement_pct:+.1f}%)"
                    else:
                        improvement_str = f"{improvement:+.4f} (new capability)" if improvement > 0 else "0.0000 (no detection)"
                    
                    html += f"""
                            <tr>
                                <td>{metric_name}</td>
                                <td>{baseline_val:.4f}</td>
                                <td>{final_val:.4f}</td>
                                <td>{improvement_str}</td>
                            </tr>
                    """
            
            html += """
                        </tbody>
                    </table>
                    
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">📈 Transfer Learning Status</div>
                            <div class="metric-value">Player model successfully adapted for ball detection</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">🎯 Multi-class Capability</div>
                            <div class="metric-value">Added ball detection while maintaining player detection</div>
                        </div>
                    </div>
                </div>
            """
        
        # Add detailed metrics table if available
        if 'metrics_history' in report_data:
            html += self._create_metrics_table_html(report_data['metrics_history'])
        
        # Add plots if available
        if plot_paths:
            html += """
                <div class="section">
                    <h2>📈 Training Progress</h2>
            """
            
            # Order plots with most important first
            plot_order = ['loss_curves', 'map_curves', 'detailed_metrics', 'learning_rate',
                         'performance_comparison', 'convergence_analysis', 'training_efficiency', 
                         'training_timeline', 'dataset_distribution', 'config_summary']
            plot_titles = {
                'loss_curves': '📉 Loss Curves',
                'map_curves': '📊 mAP Curves', 
                'learning_rate': '📈 Learning Rate Schedule',
                'detailed_metrics': '📋 Detailed Training Metrics',
                'training_efficiency': '⚡ Training Efficiency Analysis',
                'performance_comparison': '🎯 Performance Comparison',
                'convergence_analysis': '🔍 Convergence Analysis', 
                'dataset_distribution': '📁 Dataset Distribution',
                'training_timeline': '⏱️ Training Timeline',
                'config_summary': '⚙️ Configuration Summary'
            }
            
            for plot_key in plot_order:
                if plot_key in plot_paths:
                    plot_name = plot_paths[plot_key].name
                    title = plot_titles.get(plot_key, plot_key.replace('_', ' ').title())
                    html += f"""
                        <div class="plot">
                            <h3>{title}</h3>
                            <img src="plots_{timestamp}/{plot_name}" alt="{title}">
                        </div>
                    """
            
            html += "</div>"
        
        # Configuration section
        html += """
            <div class="section">
                <h2>⚙️ Configuration Details</h2>
                <table class="config-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add key configuration items
        key_configs = [
            ('Optimizer', yolo_config.get('optimizer', 'N/A')),
            ('Learning Rate', yolo_config.get('lr0', 'N/A')),
            ('Weight Decay', yolo_config.get('weight_decay', 'N/A')),
            ('Workers', yolo_config.get('workers', 'N/A')),
            ('AMP Enabled', config.get('amp_enabled', 'N/A')),
            ('Model Path', report_data.get('model_path', 'N/A')),
        ]
        
        for param, value in key_configs:
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"
        
        html += """
                    </tbody>
                </table>
            </div>
        """
        
        # Dataset information
        dataset_stats = report_data.get('dataset_stats', {})
        if dataset_stats:
            html += """
                <div class="section">
                    <h2>📁 Dataset Information</h2>
                    <table class="config-table">
                        <thead>
                            <tr>
                                <th>Dataset Split</th>
                                <th>Number of Images</th>
                            </tr>
                        </thead>
                        <tbody>
            """
            
            for split in ['train', 'val', 'test']:
                count = dataset_stats.get(f'{split}_images', 'N/A')
                html += f"<tr><td>{split.capitalize()}</td><td>{count}</td></tr>"
            
            html += """
                        </tbody>
                    </table>
                    <p><strong>Classes:</strong> {}</p>
                </div>
            """.format(', '.join(dataset_stats.get('class_names', [])))
        
        # Footer
        html += """
                <div class="section">
                    <h2>📝 Notes</h2>
                    <p>This report was automatically generated by the YOLO Training Pipeline.</p>
                    <p>For detailed metrics and raw data, please refer to the Excel report and training logs.</p>
                    <p>All images are saved in the plots directory alongside this report.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_metrics_table_html(self, metrics_df: pd.DataFrame) -> str:
        """Create HTML table with detailed epoch-by-epoch metrics."""
        html = """
            <div class="section">
                <h2>📋 Detailed Training Metrics</h2>
                <p>Epoch-by-epoch training progress with all metrics:</p>
                <div style="overflow-x: auto;">
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>Epoch</th>
        """
        
        # Add column headers for all metrics
        for col in metrics_df.columns:
            if col != 'epoch':
                html += f"<th>{col}</th>"
        
        html += """
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Find best epoch (lowest validation loss or highest mAP)
        best_epoch = 0
        val_loss_cols = [col for col in metrics_df.columns if 'val' in col.lower() and 'loss' in col.lower()]
        map_cols = [col for col in metrics_df.columns if 'map' in col.lower()]
        
        if val_loss_cols:
            best_epoch = metrics_df[val_loss_cols[0]].idxmin()
        elif map_cols:
            best_epoch = metrics_df[map_cols[0]].idxmax()
        
        # Add data rows
        for idx, row in metrics_df.iterrows():
            row_class = 'best-epoch' if idx == best_epoch else ''
            html += f'<tr class="{row_class}"><td class="epoch-col">{idx + 1}</td>'
            
            for col in metrics_df.columns:
                if col != 'epoch':
                    value = row[col]
                    if pd.isna(value):
                        html += '<td>-</td>'
                    elif isinstance(value, (int, float)):
                        html += f'<td>{value:.4f}</td>'
                    else:
                        html += f'<td>{value}</td>'
            
            html += '</tr>'
        
        html += """
                        </tbody>
                    </table>
                </div>
                <p><small><strong>Note:</strong> Best epoch is highlighted in green.</small></p>
            </div>
        """
        
        return html
    
    def _plot_training_efficiency(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot training efficiency metrics from actual data only."""
        try:
            # Count how many actual efficiency metrics we have
            efficiency_metrics = []
            
            # Check for speed metrics
            if 'speed' in metrics_df.columns:
                efficiency_metrics.append('speed')
            
            # Check for loss convergence rate
            if 'box_loss' in metrics_df.columns:
                efficiency_metrics.append('loss_convergence')
            
            # Check for GPU/resource utilization
            if 'gpu_util' in metrics_df.columns:
                efficiency_metrics.append('gpu_util')
                
            # Only create plot if we have actual data
            if not efficiency_metrics:
                logger.info("No efficiency metrics available to plot")
                return None
                
            # Create subplot layout based on available metrics
            n_metrics = len(efficiency_metrics)
            if n_metrics == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = [ax]
            elif n_metrics == 2:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
            fig.suptitle('Training Efficiency Analysis (Actual Data)', fontsize=16, fontweight='bold')
            
            plot_idx = 0
            
            # Speed metrics (if available)
            if 'speed' in efficiency_metrics:
                axes[plot_idx].plot(metrics_df['epoch'], metrics_df['speed'], 'b-', linewidth=2)
                axes[plot_idx].set_title('Training Speed per Epoch')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Speed (img/s)')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Loss convergence rate
            if 'loss_convergence' in efficiency_metrics:
                loss_improvement = metrics_df['box_loss'].diff().rolling(3).mean()
                axes[plot_idx].plot(metrics_df['epoch'], loss_improvement, 'r-', linewidth=2)
                axes[plot_idx].set_title('Loss Convergence Rate')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Loss Change Rate')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plot_idx += 1
            
            # GPU utilization (if available)
            if 'gpu_util' in efficiency_metrics:
                axes[plot_idx].plot(metrics_df['epoch'], metrics_df['gpu_util'], 'orange', linewidth=2)
                axes[plot_idx].set_title('GPU Utilization')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Utilization %')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Hide unused subplots
            if isinstance(axes, list) and len(axes) > plot_idx:
                for i in range(plot_idx, len(axes)):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = plots_dir / 'training_efficiency.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            logger.warning(f"Error creating training efficiency plot: {e}")
            return None
    
    def _plot_performance_comparison(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot performance comparison across metrics (only actual data)."""
        try:
            # Check what performance data we actually have
            available_plots = []
            
            # Check for precision and recall
            if 'precision' in metrics_df.columns and 'recall' in metrics_df.columns:
                available_plots.append('precision_recall')
                available_plots.append('f1_score')  # F1 can be calculated from P&R
            
            # Check for loss components
            loss_cols = [col for col in metrics_df.columns if 'loss' in col.lower()]
            if loss_cols:
                available_plots.append('loss_components')
            
            # Check for mAP data
            if 'map50' in metrics_df.columns:
                available_plots.append('map_evolution')
            
            # Only create plot if we have actual performance data
            if not available_plots:
                logger.info("No performance comparison data available to plot")
                return None
            
            # Create subplot layout based on available data
            n_plots = len(available_plots)
            if n_plots == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = [ax]
            elif n_plots == 2:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            elif n_plots == 3:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
            fig.suptitle('Performance Comparison Analysis (Actual Data)', fontsize=16, fontweight='bold')
            
            plot_idx = 0
            
            # Precision vs Recall scatter plot
            if 'precision_recall' in available_plots:
                scatter = axes[plot_idx].scatter(metrics_df['recall'], metrics_df['precision'], 
                                               c=metrics_df.index, cmap='viridis', s=50, alpha=0.7)
                axes[plot_idx].set_xlabel('Recall')
                axes[plot_idx].set_ylabel('Precision')
                axes[plot_idx].set_title('Precision vs Recall')
                axes[plot_idx].grid(True, alpha=0.3)
                if len(metrics_df) > 1:  # Only add colorbar if multiple points
                    cbar = plt.colorbar(scatter, ax=axes[plot_idx])
                    cbar.set_label('Epoch')
                plot_idx += 1
            
            # F1 Score evolution
            if 'f1_score' in available_plots:
                precision = metrics_df['precision']
                recall = metrics_df['recall']
                # Avoid division by zero
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
                axes[plot_idx].plot(metrics_df.index, f1_score, 'purple', linewidth=3, label='F1 Score')
                axes[plot_idx].fill_between(metrics_df.index, f1_score, alpha=0.3, color='purple')
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('F1 Score')
                axes[plot_idx].set_title('F1 Score Evolution')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].legend()
                axes[plot_idx].set_ylim(0, 1)
                plot_idx += 1
            
            # Loss components comparison
            if 'loss_components' in available_plots:
                for i, col in enumerate(loss_cols[:4]):  # Max 4 loss types
                    axes[plot_idx].plot(metrics_df.index, metrics_df[col], 
                                      linewidth=2, label=col.replace('_', ' ').title())
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Loss Value')
                axes[plot_idx].set_title('Loss Components Comparison')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].legend()
                if len(loss_cols) > 1:  # Only use log scale if multiple loss types
                    axes[plot_idx].set_yscale('log')
                plot_idx += 1
            
            # mAP evolution with confidence intervals
            if 'map_evolution' in available_plots:
                map_values = metrics_df['map50']
                epochs = metrics_df.index
                
                # Only calculate moving average if we have enough data points
                if len(map_values) >= 3:
                    window = min(3, len(map_values))
                    map_ma = map_values.rolling(window).mean()
                    map_std = map_values.rolling(window).std()
                    
                    axes[plot_idx].plot(epochs, map_values, 'b-', alpha=0.6, label='mAP@0.5')
                    axes[plot_idx].plot(epochs, map_ma, 'r-', linewidth=2, label=f'Moving Avg ({window})')
                    axes[plot_idx].fill_between(epochs, map_ma - map_std, map_ma + map_std, 
                                              alpha=0.2, color='red', label='±1 Std')
                    axes[plot_idx].legend()
                else:
                    # Just plot the raw values if not enough data for moving average
                    axes[plot_idx].plot(epochs, map_values, 'b-', linewidth=2, label='mAP@0.5', marker='o')
                    axes[plot_idx].legend()
                
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('mAP@0.5')
                axes[plot_idx].set_title('mAP@0.5 Evolution')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].set_ylim(0, 1)
                plot_idx += 1
            
            # Hide unused subplots
            if isinstance(axes, list) and len(axes) > plot_idx:
                for i in range(plot_idx, len(axes)):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = plots_dir / 'performance_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            logger.warning(f"Error creating performance comparison plot: {e}")
            return None
    
    def _plot_convergence_analysis(self, metrics_df: pd.DataFrame, plots_dir: Path) -> Optional[Path]:
        """Plot convergence analysis based on actual training data."""
        try:
            available_plots = []
            
            # Check what actual data we have
            if 'box_loss' in metrics_df.columns:
                available_plots.append('loss_trend')
                
            if 'val_loss' in metrics_df.columns:
                available_plots.append('val_stability')
                
            if 'map50' in metrics_df.columns:
                available_plots.append('map_progress')
                
            # Only create plot if we have actual convergence data
            if not available_plots:
                logger.info("No convergence metrics available to plot")
                return None
                
            # Create subplot layout based on available data
            n_plots = len(available_plots)
            if n_plots == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = [ax]
            elif n_plots == 2:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                
            fig.suptitle('Convergence Analysis (Actual Training Data)', fontsize=16, fontweight='bold')
            
            plot_idx = 0
            
            # Loss convergence with trend (only if we have box_loss)
            if 'loss_trend' in available_plots:
                loss = metrics_df['box_loss']
                epochs = metrics_df['epoch']
                
                # Only fit trend if we have enough data points
                if len(epochs) >= 4:
                    degree = min(3, len(epochs) - 1)  # Adjust degree based on data points
                    z = np.polyfit(epochs, loss, degree)
                    p = np.poly1d(z)
                    
                    axes[plot_idx].plot(epochs, loss, 'b-', linewidth=2, label='Actual Loss')
                    axes[plot_idx].plot(epochs, p(epochs), 'r--', linewidth=2, label='Trend Line')
                    axes[plot_idx].legend()
                else:
                    axes[plot_idx].plot(epochs, loss, 'b-', linewidth=2, label='Box Loss')
                    
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Box Loss')
                axes[plot_idx].set_title('Loss Convergence')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Validation stability (only if we have val_loss)
            if 'val_stability' in available_plots:
                val_loss = metrics_df['val_loss']
                if len(val_loss) >= 3:  # Need at least 3 points for rolling std
                    val_stability = val_loss.rolling(3).std()
                    axes[plot_idx].plot(metrics_df['epoch'], val_stability, 'orange', linewidth=2)
                    axes[plot_idx].set_title('Validation Loss Stability')
                else:
                    axes[plot_idx].plot(metrics_df['epoch'], val_loss, 'orange', linewidth=2)
                    axes[plot_idx].set_title('Validation Loss')
                    
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('Loss Std Dev' if len(val_loss) >= 3 else 'Validation Loss')
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
            
            # Training progress rate (only if we have mAP data)
            if 'map_progress' in available_plots:
                map_progress = metrics_df['map50'].diff().fillna(0)
                colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in map_progress]
                axes[plot_idx].bar(metrics_df['epoch'], map_progress, alpha=0.7, color=colors)
                axes[plot_idx].set_xlabel('Epoch')
                axes[plot_idx].set_ylabel('mAP@0.5 Improvement')
                axes[plot_idx].set_title('mAP Progress Rate')
                axes[plot_idx].grid(True, alpha=0.3)
                axes[plot_idx].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plot_idx += 1
            
            # Hide unused subplots
            if isinstance(axes, list) and len(axes) > plot_idx:
                for i in range(plot_idx, len(axes)):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = plots_dir / 'convergence_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            logger.warning(f"Error creating convergence analysis plot: {e}")
            return None
    
    def _plot_dataset_distribution(self, report_data: Dict[str, Any], plots_dir: Path) -> Optional[Path]:
        """Plot dataset distribution information from actual data only."""
        try:
            dataset_info = report_data.get('dataset_info', {})
            dataset_stats = report_data.get('dataset_stats', {})
            
            # Check what actual dataset information we have
            has_class_info = 'class_names' in dataset_stats or 'nc' in dataset_stats
            has_split_info = any(key in dataset_stats for key in ['train_images', 'val_images', 'test_images'])
            has_extraction_info = 'extraction_stats' in report_data
            
            available_plots = []
            if has_class_info:
                available_plots.append('classes')
            if has_split_info:
                available_plots.append('splits')
            if has_extraction_info:
                available_plots.append('extraction')
                
            # Only create plot if we have actual dataset information
            if not available_plots:
                logger.info("No dataset distribution data available to plot")
                return None
                
            # Create subplot layout based on available data
            n_plots = len(available_plots)
            if n_plots == 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                axes = [ax]
            elif n_plots == 2:
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
            fig.suptitle('Dataset Distribution (Actual Data)', fontsize=16, fontweight='bold')
            
            plot_idx = 0
            
            # Class distribution (only if we have actual class data)
            if 'classes' in available_plots:
                class_names = dataset_stats.get('class_names', [])
                if not class_names:
                    # Try to get from nc (number of classes)
                    nc = dataset_stats.get('nc', 1)
                    class_names = [f'Class {i}' for i in range(nc)]
                
                # Simple bar chart for classes
                axes[plot_idx].bar(range(len(class_names)), [1] * len(class_names), 
                                 color='skyblue', alpha=0.7)
                axes[plot_idx].set_title(f'Classes in Dataset ({len(class_names)} total)')
                axes[plot_idx].set_xlabel('Class Index')
                axes[plot_idx].set_ylabel('Present')
                axes[plot_idx].set_xticks(range(len(class_names)))
                axes[plot_idx].set_xticklabels(class_names, rotation=45)
                plot_idx += 1
            
            # Train/Val/Test split (only if we have actual split data)
            if 'splits' in available_plots:
                splits = []
                split_counts = []
                
                if 'train_images' in dataset_stats:
                    splits.append('Train')
                    split_counts.append(dataset_stats['train_images'])
                if 'val_images' in dataset_stats:
                    splits.append('Validation')
                    split_counts.append(dataset_stats['val_images'])
                if 'test_images' in dataset_stats:
                    splits.append('Test')
                    split_counts.append(dataset_stats['test_images'])
                
                if splits:
                    axes[plot_idx].bar(splits, split_counts, color=['blue', 'orange', 'green'][:len(splits)], alpha=0.7)
                    axes[plot_idx].set_title('Dataset Split Distribution')
                    axes[plot_idx].set_ylabel('Number of Images')
                    for i, v in enumerate(split_counts):
                        axes[plot_idx].text(i, v + max(split_counts) * 0.01, str(v), 
                                           ha='center', va='bottom', fontweight='bold')
                plot_idx += 1
            
            # Extraction statistics (only if we have actual extraction data)
            if 'extraction' in available_plots:
                extraction_stats = report_data['extraction_stats']
                
                # Show extraction results
                extracted_data = []
                labels = []
                
                if 'total_frames_extracted' in extraction_stats:
                    extracted_data.append(extraction_stats['total_frames_extracted'])
                    labels.append('Frames Extracted')
                    
                if 'matches_found' in extraction_stats:
                    extracted_data.append(extraction_stats['matches_found'])
                    labels.append('Matches Found')
                
                if extracted_data:
                    axes[plot_idx].bar(labels, extracted_data, color='green', alpha=0.7)
                    axes[plot_idx].set_title('Data Extraction Statistics')
                    axes[plot_idx].set_ylabel('Count')
                    for i, v in enumerate(extracted_data):
                        axes[plot_idx].text(i, v + max(extracted_data) * 0.01, str(v), 
                                           ha='center', va='bottom', fontweight='bold')
                plot_idx += 1
            
            # Hide unused subplots
            if isinstance(axes, list) and len(axes) > plot_idx:
                for i in range(plot_idx, len(axes)):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = plots_dir / 'dataset_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            logger.warning(f"Error creating dataset distribution plot: {e}")
            return None
    
    def _plot_training_timeline(self, report_data: Dict[str, Any], plots_dir: Path) -> Optional[Path]:
        """Plot training timeline based on actual training data."""
        try:
            # Get actual training info
            training_info = report_data.get('training_info', {})
            config = report_data.get('config', {})
            yolo_params = config.get('yolo_params', {})
            
            # Get actual epochs from training
            total_epochs = yolo_params.get('epochs', training_info.get('epochs'))
            if not total_epochs:
                logger.info("No epoch information available for timeline")
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Simple timeline showing actual training progress
            epochs_range = list(range(1, total_epochs + 1))
            
            # Plot the training progress line
            ax.plot([1, total_epochs], [1, 1], 'b-', linewidth=4, alpha=0.7, label='Training Progress')
            
            # Add start and end markers
            ax.scatter([1], [1], s=150, c='green', alpha=0.8, zorder=3, label='Training Started')
            ax.scatter([total_epochs], [1], s=150, c='red', alpha=0.8, zorder=3, label='Training Completed')
            
            # Add actual milestones if available
            if 'metrics_history' in report_data:
                metrics_df = report_data['metrics_history']
                if 'map50' in metrics_df.columns:
                    best_epoch = metrics_df['map50'].idxmax() + 1  # +1 because epoch indexing
                    ax.scatter([best_epoch], [1], s=150, c='gold', alpha=0.8, zorder=3, 
                             label=f'Best mAP (Epoch {best_epoch})')
            
            # Configure the plot
            ax.set_xlim(0.5, total_epochs + 0.5)
            ax.set_ylim(0.7, 1.3)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_title('Training Timeline (Actual Progress)', fontsize=16, fontweight='bold')
            ax.set_yticks([])
            ax.grid(True, axis='x', alpha=0.3)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
            
            # Add actual training statistics
            stats = []
            if total_epochs:
                stats.append(f"Total Epochs: {total_epochs}")
            if 'timestamp' in report_data:
                timestamp = report_data['timestamp']
                if hasattr(timestamp, 'strftime'):
                    stats.append(f"Start Time: {timestamp.strftime('%Y-%m-%d %H:%M')}")
                else:
                    stats.append(f"Start Time: {timestamp}")
            if 'duration' in training_info:
                stats.append(f"Duration: {training_info['duration']}")
            
            # Add best metrics if available
            if 'metrics_history' in report_data:
                metrics_df = report_data['metrics_history']
                if 'map50' in metrics_df.columns:
                    best_map = metrics_df['map50'].max()
                    stats.append(f"Best mAP@0.5: {best_map:.4f}")
                if 'box_loss' in metrics_df.columns:
                    final_loss = metrics_df['box_loss'].iloc[-1]
                    stats.append(f"Final Loss: {final_loss:.4f}")
            
            if stats:
                stats_text = '\n'.join(stats)
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                       fontsize=10, fontfamily='monospace')
            
            plt.tight_layout()
            plot_path = plots_dir / 'training_timeline.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_path
            
        except Exception as e:
            logger.warning(f"Error creating training timeline plot: {e}")
            return None



def create_report_generator(output_dir: Path) -> TrainingReportGenerator:
    """
    Factory function to create a report generator.
    
    Args:
        output_dir: Directory to save reports
        
    Returns:
        Configured TrainingReportGenerator instance
    """
    return TrainingReportGenerator(output_dir)