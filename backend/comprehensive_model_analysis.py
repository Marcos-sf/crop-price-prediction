#!/usr/bin/env python3
"""
Comprehensive Model Performance Analysis
This script analyzes all 3 models (XGBoost, LSTM, Ensemble) and displays:
- Model accuracies
- Confusion matrices
- Performance plots
- Statistical analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

def load_enhanced_models_data():
    """Load data from enhanced models and analyze performance"""
    
    # Define crop-mandi combinations
    crop_mandi_combinations = [
        ('arecanut', 'sirsi'), ('arecanut', 'yellapur'), ('arecanut', 'siddapur'),
        ('arecanut', 'shimoga'), ('arecanut', 'sagar'), ('arecanut', 'kumta'),
        ('coconut', 'bangalore'), ('coconut', 'arasikere'), ('coconut', 'channarayapatna'),
        ('coconut', 'ramanagara'), ('coconut', 'sira'), ('coconut', 'tumkur')
    ]
    
    results = []
    
    for crop, mandi in crop_mandi_combinations:
        try:
            # Check if enhanced models exist
            xgb_path = f"app/data/processed/xgb_enhanced_{crop}_{mandi}.joblib"
            lstm_path = f"app/data/processed/lstm_enhanced_{crop}_{mandi}.h5"
            ensemble_path = f"app/data/processed/ensemble_enhanced_{crop}_{mandi}.joblib"
            
            result = {
                'crop': crop,
                'mandi': mandi,
                'xgb_enhanced': os.path.exists(xgb_path),
                'lstm_enhanced': os.path.exists(lstm_path),
                'ensemble_enhanced': os.path.exists(ensemble_path)
            }
            
            # Load ensemble metadata if available
            if os.path.exists(ensemble_path):
                try:
                    ensemble_metadata = joblib.load(ensemble_path)
                    if 'xgb_accuracy' in ensemble_metadata:
                        result['xgb_accuracy'] = ensemble_metadata['xgb_accuracy']
                    if 'lstm_accuracy' in ensemble_metadata:
                        result['lstm_accuracy'] = ensemble_metadata['lstm_accuracy']
                    if 'ensemble_accuracy' in ensemble_metadata:
                        result['ensemble_accuracy'] = ensemble_metadata['ensemble_accuracy']
                    if 'weights' in ensemble_metadata:
                        result['xgb_weight'] = ensemble_metadata['weights'][0]
                        result['lstm_weight'] = ensemble_metadata['weights'][1]
                except Exception as e:
                    print(f"⚠️ Error loading ensemble metadata for {crop}_{mandi}: {e}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error analyzing {crop}_{mandi}: {e}")
    
    return pd.DataFrame(results)

def create_performance_plots(enhanced_models_df):
    """Create comprehensive performance visualization"""
    
    if 'ensemble_accuracy' not in enhanced_models_df.columns:
        print("⚠️ No accuracy data available for plotting")
        return
    
    # Filter models with accuracy data
    accuracy_models = enhanced_models_df[enhanced_models_df['ensemble_accuracy'].notna()].copy()
    
    if len(accuracy_models) == 0:
        print("⚠️ No accuracy data available for plotting")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('🎯 Model Performance Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # 1. Accuracy Comparison Bar Chart
    ax1 = axes[0, 0]
    x_pos = np.arange(len(accuracy_models))
    width = 0.25
    
    xgb_acc = accuracy_models['xgb_accuracy'].fillna(0)
    lstm_acc = accuracy_models['lstm_accuracy'].fillna(0)
    ensemble_acc = accuracy_models['ensemble_accuracy'].fillna(0)
    
    ax1.bar(x_pos - width, xgb_acc, width, label='XGBoost', alpha=0.8, color='#FF6B6B')
    ax1.bar(x_pos, lstm_acc, width, label='LSTM', alpha=0.8, color='#4ECDC4')
    ax1.bar(x_pos + width, ensemble_acc, width, label='Ensemble', alpha=0.8, color='#45B7D1')
    
    ax1.set_xlabel('Crop-Mandi Combinations', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{row['crop'][:3].title()}-{row['mandi'][:3].title()}" 
                          for _, row in accuracy_models.iterrows()], rotation=45)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. 85% Threshold Achievement
    ax2 = axes[0, 1]
    thresholds = ['XGBoost ≥85%', 'LSTM ≥85%', 'Ensemble ≥85%']
    counts = [
        len(accuracy_models[accuracy_models['xgb_accuracy'] >= 85]),
        len(accuracy_models[accuracy_models['lstm_accuracy'] >= 85]),
        len(accuracy_models[accuracy_models['ensemble_accuracy'] >= 85])
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax2.bar(thresholds, counts, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Models', fontsize=12)
    ax2.set_title('Models Achieving 85%+ Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # 3. Accuracy Distribution
    ax3 = axes[1, 0]
    all_accuracies = []
    labels = []
    
    for acc, label in [(xgb_acc, 'XGBoost'), (lstm_acc, 'LSTM'), (ensemble_acc, 'Ensemble')]:
        all_accuracies.extend(acc)
        labels.extend([label] * len(acc))
    
    acc_df = pd.DataFrame({'Accuracy': all_accuracies, 'Model': labels})
    
    sns.boxplot(data=acc_df, x='Model', y='Accuracy', ax=ax3, palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('Accuracy Distribution by Model Type', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance Heatmap
    ax4 = axes[1, 1]
    heatmap_data = accuracy_models[['xgb_accuracy', 'lstm_accuracy', 'ensemble_accuracy']].fillna(0)
    heatmap_data.index = [f"{row['crop'][:3].title()}-{row['mandi'][:3].title()}" 
                           for _, row in accuracy_models.iterrows()]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax4, cbar_kws={'label': 'Accuracy %'})
    ax4.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Model Type', fontsize=12)
    ax4.set_ylabel('Crop-Mandi Combination', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Performance dashboard saved as: model_performance_dashboard.png")

def create_confusion_matrix_analysis(enhanced_models_df):
    """Create confusion matrix analysis for classification tasks"""
    
    print("\n🔍 CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    if 'ensemble_accuracy' not in enhanced_models_df.columns:
        print("⚠️ No accuracy data available for confusion matrix analysis")
        return
    
    accuracy_models = enhanced_models_df[enhanced_models_df['ensemble_accuracy'].notna()].copy()
    
    if len(accuracy_models) == 0:
        print("⚠️ No accuracy data available for confusion matrix analysis")
        return
    
    # Create confusion matrix visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Confusion Matrix Analysis (85% Threshold)', fontsize=16, fontweight='bold')
    
    # Define threshold for classification
    threshold = 85
    
    # XGBoost Confusion Matrix
    ax1 = axes[0]
    xgb_acc = accuracy_models['xgb_accuracy'].fillna(0)
    xgb_binary = (xgb_acc >= threshold).astype(int)
    
    # Create confusion matrix data
    xgb_cm = confusion_matrix([1] * len(xgb_binary), xgb_binary)
    
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Below 85%', 'Above 85%'], 
                yticklabels=['Below 85%', 'Above 85%'])
    ax1.set_title(f'XGBoost Performance\n(≥{threshold}% Accuracy)', fontweight='bold')
    ax1.set_xlabel('Predicted', fontweight='bold')
    ax1.set_ylabel('Actual', fontweight='bold')
    
    # LSTM Confusion Matrix
    ax2 = axes[1]
    lstm_acc = accuracy_models['lstm_accuracy'].fillna(0)
    lstm_binary = (lstm_acc >= threshold).astype(int)
    
    lstm_cm = confusion_matrix([1] * len(lstm_binary), lstm_binary)
    
    sns.heatmap(lstm_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Below 85%', 'Above 85%'], 
                yticklabels=['Below 85%', 'Above 85%'])
    ax2.set_title(f'LSTM Performance\n(≥{threshold}% Accuracy)', fontweight='bold')
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_ylabel('Actual', fontweight='bold')
    
    # Ensemble Confusion Matrix
    ax3 = axes[2]
    ensemble_acc = accuracy_models['ensemble_accuracy'].fillna(0)
    ensemble_binary = (ensemble_acc >= threshold).astype(int)
    
    ensemble_cm = confusion_matrix([1] * len(ensemble_binary), ensemble_binary)
    
    sns.heatmap(ensemble_cm, annot=True, fmt='d', cmap='Reds', ax=ax3,
                xticklabels=['Below 85%', 'Above 85%'], 
                yticklabels=['Below 85%', 'Above 85%'])
    ax3.set_title(f'Ensemble Performance\n(≥{threshold}% Accuracy)', fontweight='bold')
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Actual', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Confusion matrix analysis saved as: confusion_matrix_analysis.png")

def detailed_performance_analysis(enhanced_models_df):
    """Perform detailed performance analysis"""
    
    print("\n📊 DETAILED PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if 'ensemble_accuracy' not in enhanced_models_df.columns:
        print("⚠️ No accuracy data available for detailed analysis")
        return
    
    accuracy_models = enhanced_models_df[enhanced_models_df['ensemble_accuracy'].notna()].copy()
    
    if len(accuracy_models) == 0:
        print("⚠️ No accuracy data available for detailed analysis")
        return
    
    # XGBoost Analysis
    print("🚀 XGBOOST ANALYSIS:")
    print("-" * 40)
    xgb_acc = accuracy_models['xgb_accuracy'].fillna(0)
    print(f"Total Models: {len(xgb_acc)}")
    print(f"Average Accuracy: {xgb_acc.mean():.2f}%")
    print(f"Best Accuracy: {xgb_acc.max():.2f}%")
    print(f"Models ≥85%: {len(xgb_acc[xgb_acc >= 85])}/{len(xgb_acc)} ({len(xgb_acc[xgb_acc >= 85])/len(xgb_acc)*100:.1f}%)")
    print(f"Models ≥90%: {len(xgb_acc[xgb_acc >= 90])}/{len(xgb_acc)} ({len(xgb_acc[xgb_acc >= 90])/len(xgb_acc)*100:.1f}%)")
    
    # LSTM Analysis
    print("\n🧠 LSTM ANALYSIS:")
    print("-" * 40)
    lstm_acc = accuracy_models['lstm_accuracy'].fillna(0)
    print(f"Total Models: {len(lstm_acc)}")
    print(f"Average Accuracy: {lstm_acc.mean():.2f}%")
    print(f"Best Accuracy: {lstm_acc.max():.2f}%")
    print(f"Models ≥85%: {len(lstm_acc[lstm_acc >= 85])}/{len(lstm_acc)} ({len(lstm_acc[lstm_acc >= 85])/len(lstm_acc)*100:.1f}%)")
    print(f"Models ≥90%: {len(lstm_acc[lstm_acc >= 90])}/{len(lstm_acc)} ({len(lstm_acc[lstm_acc >= 90])/len(lstm_acc)*100:.1f}%)")
    
    # Ensemble Analysis
    print("\n🔗 ENSEMBLE ANALYSIS:")
    print("-" * 40)
    ensemble_acc = accuracy_models['ensemble_accuracy'].fillna(0)
    print(f"Total Models: {len(ensemble_acc)}")
    print(f"Average Accuracy: {ensemble_acc.mean():.2f}%")
    print(f"Best Accuracy: {ensemble_acc.max():.2f}%")
    print(f"Models ≥85%: {len(ensemble_acc[ensemble_acc >= 85])}/{len(ensemble_acc)} ({len(ensemble_acc[ensemble_acc >= 85])/len(ensemble_acc)*100:.1f}%)")
    print(f"Models ≥90%: {len(ensemble_acc[ensemble_acc >= 90])}/{len(ensemble_acc)} ({len(ensemble_acc[ensemble_acc >= 90])/len(ensemble_acc)*100:.1f}%)")
    
    # Overall Summary
    print("\n📊 OVERALL SUMMARY:")
    print("-" * 40)
    total_models = len(accuracy_models)
    
    # Count models achieving 85%+ in any category
    any_85_plus = len(accuracy_models[
        (accuracy_models['xgb_accuracy'] >= 85) | 
        (accuracy_models['lstm_accuracy'] >= 85) | 
        (accuracy_models['ensemble_accuracy'] >= 85)
    ])
    
    print(f"Total Crop-Mandi Combinations: {total_models}")
    print(f"Combinations with ≥85% accuracy in ANY model: {any_85_plus}/{total_models} ({any_85_plus/total_models*100:.1f}%)")
    
    # Best performing combinations
    print("\n🏆 TOP PERFORMING COMBINATIONS:")
    top_performers = accuracy_models[
        (accuracy_models['xgb_accuracy'] >= 85) | 
        (accuracy_models['lstm_accuracy'] >= 85) | 
        (accuracy_models['ensemble_accuracy'] >= 85)
    ].copy()
    
    if len(top_performers) > 0:
        for _, row in top_performers.iterrows():
            print(f"  {row['crop'].title()} - {row['mandi'].title()}")
            if row['xgb_accuracy'] >= 85:
                print(f"    🚀 XGBoost: {row['xgb_accuracy']:.1f}%")
            if row['lstm_accuracy'] >= 85:
                print(f"    🧠 LSTM: {row['lstm_accuracy']:.1f}%")
            if row['ensemble_accuracy'] >= 85:
                print(f"    🔗 Ensemble: {row['ensemble_accuracy']:.1f}%")
            print()
    else:
        print("  No models currently achieving 85%+ accuracy")

def save_analysis_results(enhanced_models_df):
    """Save all analysis results to CSV files"""
    
    print("\n💾 SAVING ANALYSIS RESULTS")
    print("=" * 40)
    
    # Save enhanced models status
    enhanced_models_df.to_csv('enhanced_models_comprehensive_analysis.csv', index=False)
    print("✅ Enhanced models analysis saved to: enhanced_models_comprehensive_analysis.csv")
    
    # Save accuracy summary
    if 'ensemble_accuracy' in enhanced_models_df.columns:
        accuracy_models = enhanced_models_df[enhanced_models_df['ensemble_accuracy'].notna()].copy()
        
        # Create summary dataframe
        summary_data = []
        for _, row in accuracy_models.iterrows():
            summary_data.append({
                'crop': row['crop'],
                'mandi': row['mandi'],
                'xgb_accuracy': row.get('xgb_accuracy', 0),
                'lstm_accuracy': row.get('lstm_accuracy', 0),
                'ensemble_accuracy': row.get('ensemble_accuracy', 0),
                'xgb_weight': row.get('xgb_weight', 'N/A'),
                'lstm_weight': row.get('lstm_weight', 'N/A'),
                'xgb_85_plus': row.get('xgb_accuracy', 0) >= 85,
                'lstm_85_plus': row.get('lstm_accuracy', 0) >= 85,
                'ensemble_85_plus': row.get('ensemble_accuracy', 0) >= 85,
                'any_85_plus': (row.get('xgb_accuracy', 0) >= 85) or 
                               (row.get('lstm_accuracy', 0) >= 85) or 
                               (row.get('ensemble_accuracy', 0) >= 85)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('model_performance_summary.csv', index=False)
        print("✅ Model performance summary saved to: model_performance_summary.csv")
        
        # Display summary
        print("\n📋 PERFORMANCE SUMMARY:")
        print(f"Total Models Analyzed: {len(summary_df)}")
        print(f"Models with ≥85% accuracy in any category: {summary_df['any_85_plus'].sum()}/{len(summary_df)}")
        print(f"XGBoost ≥85%: {summary_df['xgb_85_plus'].sum()}/{len(summary_df)}")
        print(f"LSTM ≥85%: {summary_df['lstm_85_plus'].sum()}/{len(summary_df)}")
        print(f"Ensemble ≥85%: {summary_df['ensemble_85_plus'].sum()}/{len(summary_df)}")
    
    print("\n✅ Analysis complete! All results saved.")

def main():
    """Main function to run comprehensive analysis"""
    
    print("🎯 COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    print("Analyzing XGBoost, LSTM, and Ensemble models...")
    
    # Load enhanced models data
    enhanced_models_df = load_enhanced_models_data()
    
    if enhanced_models_df.empty:
        print("❌ No enhanced models found!")
        return
    
    print(f"✅ Loaded data for {len(enhanced_models_df)} crop-mandi combinations")
    
    # Display current status
    print(f"\n📊 ENHANCED MODEL AVAILABILITY:")
    print(f"XGBoost Enhanced: {enhanced_models_df['xgb_enhanced'].sum()}/{len(enhanced_models_df)}")
    print(f"LSTM Enhanced: {enhanced_models_df['lstm_enhanced'].sum()}/{len(enhanced_models_df)}")
    print(f"Ensemble Enhanced: {enhanced_models_df['ensemble_enhanced'].sum()}/{len(enhanced_models_df)}")
    
    # Create performance plots
    create_performance_plots(enhanced_models_df)
    
    # Create confusion matrix analysis
    create_confusion_matrix_analysis(enhanced_models_df)
    
    # Perform detailed analysis
    detailed_performance_analysis(enhanced_models_df)
    
    # Save results
    save_analysis_results(enhanced_models_df)
    
    print("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📁 Check the generated files and plots for detailed results.")

if __name__ == "__main__":
    main()

