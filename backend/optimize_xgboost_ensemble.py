#!/usr/bin/env python3
"""
Optimize XGBoost and Ensemble Performance
This script focuses on optimizing and displaying XGBoost and Ensemble models only.
LSTM is kept in the background but not displayed in graphs or metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)

def load_xgboost_ensemble_data():
    """Load data focusing only on XGBoost and Ensemble models"""
    
    print("🚀 LOADING XGBOOST & ENSEMBLE DATA")
    print("=" * 60)
    
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
            ensemble_path = f"app/data/processed/ensemble_enhanced_{crop}_{mandi}.joblib"
            
            result = {
                'crop': crop,
                'mandi': mandi,
                'xgb_enhanced': os.path.exists(xgb_path),
                'ensemble_enhanced': os.path.exists(ensemble_path)
            }
            
            # Load ensemble metadata if available
            if os.path.exists(ensemble_path):
                try:
                    ensemble_metadata = joblib.load(ensemble_path)
                    if 'xgb_accuracy' in ensemble_metadata:
                        result['xgb_accuracy'] = ensemble_metadata['xgb_accuracy']
                    if 'ensemble_accuracy' in ensemble_metadata:
                        result['ensemble_accuracy'] = ensemble_metadata['ensemble_accuracy']
                    if 'weights' in ensemble_metadata:
                        result['xgb_weight'] = ensemble_metadata['weights'][0]
                        result['lstm_weight'] = ensemble_metadata['weights'][1]  # Keep for reference only
                except Exception as e:
                    print(f"⚠️ Error loading ensemble metadata for {crop}_{mandi}: {e}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error analyzing {crop}_{mandi}: {e}")
    
    return pd.DataFrame(results)

def create_xgboost_ensemble_dashboard(enhanced_models_df):
    """Create comprehensive dashboard for XGBoost and Ensemble only"""
    
    if 'ensemble_accuracy' not in enhanced_models_df.columns:
        print("⚠️ No accuracy data available for plotting")
        return
    
    # Filter models with accuracy data
    accuracy_models = enhanced_models_df[enhanced_models_df['ensemble_accuracy'].notna()].copy()
    
    if len(accuracy_models) == 0:
        print("⚠️ No accuracy data available for plotting")
        return
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    fig.suptitle('🎯 XGBoost & Ensemble Performance Dashboard (LSTM Hidden)', fontsize=24, fontweight='bold')
    
    # 1. XGBoost vs Ensemble Accuracy Comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(accuracy_models))
    width = 0.35
    
    xgb_acc = accuracy_models['xgb_accuracy'].fillna(0)
    ensemble_acc = accuracy_models['ensemble_accuracy'].fillna(0)
    
    bars1 = ax1.bar(x_pos - width/2, xgb_acc, width, label='XGBoost', alpha=0.8, color='#FF6B6B')
    bars2 = ax1.bar(x_pos + width/2, ensemble_acc, width, label='Ensemble', alpha=0.8, color='#45B7D1')
    
    ax1.set_xlabel('Crop-Mandi Combinations', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('XGBoost vs Ensemble Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{row['crop'][:3].title()}-{row['mandi'][:3].title()}" 
                          for _, row in accuracy_models.iterrows()], rotation=45)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. 85% Threshold Achievement
    ax2 = axes[0, 1]
    thresholds = ['XGBoost ≥85%', 'Ensemble ≥85%']
    counts = [
        len(accuracy_models[accuracy_models['xgb_accuracy'] >= 85]),
        len(accuracy_models[accuracy_models['ensemble_accuracy'] >= 85])
    ]
    
    colors = ['#FF6B6B', '#45B7D1']
    bars = ax2.bar(thresholds, counts, color=colors, alpha=0.8)
    ax2.set_ylabel('Number of Models', fontsize=12)
    ax2.set_title('Models Achieving 85%+ Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # 3. Performance Heatmap
    ax3 = axes[0, 2]
    heatmap_data = accuracy_models[['xgb_accuracy', 'ensemble_accuracy']].fillna(0)
    heatmap_data.index = [f"{row['crop'][:3].title()}-{row['mandi'][:3].title()}" 
                           for _, row in accuracy_models.iterrows()]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax3, cbar_kws={'label': 'Accuracy %'})
    ax3.set_title('XGBoost & Ensemble Performance Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model Type', fontsize=12)
    ax3.set_ylabel('Crop-Mandi Combination', fontsize=12)
    
    # 4. Accuracy Distribution
    ax4 = axes[1, 0]
    all_accuracies = []
    labels = []
    
    for acc, label in [(xgb_acc, 'XGBoost'), (ensemble_acc, 'Ensemble')]:
        all_accuracies.extend(acc)
        labels.extend([label] * len(acc))
    
    acc_df = pd.DataFrame({'Accuracy': all_accuracies, 'Model': labels})
    
    sns.boxplot(data=acc_df, x='Model', y='Accuracy', ax=ax4, palette=['#FF6B6B', '#45B7D1'])
    ax4.set_title('Accuracy Distribution by Model Type', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Top Performers Ranking
    ax5 = axes[1, 1]
    # Get top 8 performing combinations
    top_performers = accuracy_models.nlargest(8, 'xgb_accuracy')
    
    y_pos = np.arange(len(top_performers))
    xgb_performance = top_performers['xgb_accuracy'].fillna(0)
    
    bars = ax5.barh(y_pos, xgb_performance, color='#FF6B6B', alpha=0.8)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([f"{row['crop'][:3].title()}-{row['mandi'][:3].title()}" 
                          for _, row in top_performers.iterrows()])
    ax5.set_xlabel('Accuracy (%)', fontsize=12)
    ax5.set_title('Top 8 XGBoost Performers', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, xgb_performance)):
        ax5.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 6. 85%+ Achievement Summary
    ax6 = axes[1, 2]
    # Calculate percentages
    total_models = len(accuracy_models)
    xgb_85_plus = len(accuracy_models[accuracy_models['xgb_accuracy'] >= 85])
    ensemble_85_plus = len(accuracy_models[accuracy_models['ensemble_accuracy'] >= 85])
    
    categories = ['XGBoost ≥85%', 'Ensemble ≥85%', 'Both ≥85%', 'Neither ≥85%']
    counts = [
        xgb_85_plus,
        ensemble_85_plus,
        len(accuracy_models[(accuracy_models['xgb_accuracy'] >= 85) & (accuracy_models['ensemble_accuracy'] >= 85)]),
        len(accuracy_models[(accuracy_models['xgb_accuracy'] < 85) & (accuracy_models['ensemble_accuracy'] < 85)])
    ]
    
    colors = ['#FF6B6B', '#45B7D1', '#4ECDC4', '#95A5A6']
    wedges, texts, autotexts = ax6.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('85%+ Accuracy Achievement Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('xgboost_ensemble_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 XGBoost & Ensemble dashboard saved as: xgboost_ensemble_dashboard.png")

def create_confusion_matrix_xgboost_ensemble(enhanced_models_df):
    """Create confusion matrix analysis for XGBoost and Ensemble only"""
    
    print("\n🔍 CONFUSION MATRIX ANALYSIS (XGBoost & Ensemble Only)")
    print("=" * 70)
    
    if 'ensemble_accuracy' not in enhanced_models_df.columns:
        print("⚠️ No accuracy data available for confusion matrix analysis")
        return
    
    accuracy_models = enhanced_models_df[enhanced_models_df['ensemble_accuracy'].notna()].copy()
    
    if len(accuracy_models) == 0:
        print("⚠️ No accuracy data available for confusion matrix analysis")
        return
    
    # Create confusion matrix visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Confusion Matrix Analysis - 85% Threshold (XGBoost & Ensemble Only)', fontsize=16, fontweight='bold')
    
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
    
    # Ensemble Confusion Matrix
    ax2 = axes[1]
    ensemble_acc = accuracy_models['ensemble_accuracy'].fillna(0)
    ensemble_binary = (ensemble_acc >= threshold).astype(int)
    
    ensemble_cm = confusion_matrix([1] * len(ensemble_binary), ensemble_binary)
    
    sns.heatmap(ensemble_cm, annot=True, fmt='d', cmap='Reds', ax=ax2,
                xticklabels=['Below 85%', 'Above 85%'], 
                yticklabels=['Below 85%', 'Above 85%'])
    ax2.set_title(f'Ensemble Performance\n(≥{threshold}% Accuracy)', fontweight='bold')
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_ylabel('Actual', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('xgboost_ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("💾 Confusion matrix analysis saved as: xgboost_ensemble_confusion_matrix.png")

def detailed_xgboost_ensemble_analysis(enhanced_models_df):
    """Perform detailed analysis of XGBoost and Ensemble models only"""
    
    print("\n📊 DETAILED XGBOOST & ENSEMBLE ANALYSIS")
    print("=" * 70)
    
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
    print(f"Models ≥95%: {len(xgb_acc[xgb_acc >= 95])}/{len(xgb_acc)} ({len(xgb_acc[xgb_acc >= 95])/len(xgb_acc)*100:.1f}%)")
    
    # Ensemble Analysis
    print("\n🔗 ENSEMBLE ANALYSIS:")
    print("-" * 40)
    ensemble_acc = accuracy_models['ensemble_accuracy'].fillna(0)
    print(f"Total Models: {len(ensemble_acc)}")
    print(f"Average Accuracy: {ensemble_acc.mean():.2f}%")
    print(f"Best Accuracy: {ensemble_acc.max():.2f}%")
    print(f"Models ≥85%: {len(ensemble_acc[ensemble_acc >= 85])}/{len(ensemble_acc)} ({len(ensemble_acc[ensemble_acc >= 85])/len(ensemble_acc)*100:.1f}%)")
    print(f"Models ≥90%: {len(ensemble_acc[ensemble_acc >= 90])}/{len(ensemble_acc)} ({len(ensemble_acc[ensemble_acc >= 90])/len(ensemble_acc)*100:.1f}%)")
    print(f"Models ≥95%: {len(ensemble_acc[ensemble_acc >= 95])}/{len(ensemble_acc)} ({len(ensemble_acc[ensemble_acc >= 95])/len(ensemble_acc)*100:.1f}%)")
    
    # Overall Summary
    print("\n📊 OVERALL SUMMARY:")
    print("-" * 40)
    total_models = len(accuracy_models)
    
    # Count models achieving 85%+ in any category
    any_85_plus = len(accuracy_models[
        (accuracy_models['xgb_accuracy'] >= 85) | 
        (accuracy_models['ensemble_accuracy'] >= 85)
    ])
    
    print(f"Total Crop-Mandi Combinations: {total_models}")
    print(f"Combinations with ≥85% accuracy in ANY model: {any_85_plus}/{total_models} ({any_85_plus/total_models*100:.1f}%)")
    
    # Best performing combinations
    print("\n🏆 TOP PERFORMING COMBINATIONS:")
    top_performers = accuracy_models[
        (accuracy_models['xgb_accuracy'] >= 85) | 
        (accuracy_models['ensemble_accuracy'] >= 85)
    ].copy()
    
    if len(top_performers) > 0:
        for _, row in top_performers.iterrows():
            print(f"  {row['crop'].title()} - {row['mandi'].title()}")
            if row['xgb_accuracy'] >= 85:
                print(f"    🚀 XGBoost: {row['xgb_accuracy']:.1f}%")
            if row['ensemble_accuracy'] >= 85:
                print(f"    🔗 Ensemble: {row['ensemble_accuracy']:.1f}%")
            print()
    else:
        print("  No models currently achieving 85%+ accuracy")

def save_xgboost_ensemble_results(enhanced_models_df):
    """Save XGBoost and Ensemble analysis results"""
    
    print("\n💾 SAVING XGBOOST & ENSEMBLE RESULTS")
    print("=" * 50)
    
    # Save enhanced models status
    enhanced_models_df.to_csv('xgboost_ensemble_analysis.csv', index=False)
    print("✅ XGBoost & Ensemble analysis saved to: xgboost_ensemble_analysis.csv")
    
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
                'ensemble_accuracy': row.get('ensemble_accuracy', 0),
                'xgb_weight': row.get('xgb_weight', 'N/A'),
                'xgb_85_plus': row.get('xgb_accuracy', 0) >= 85,
                'ensemble_85_plus': row.get('ensemble_accuracy', 0) >= 85,
                'any_85_plus': (row.get('xgb_accuracy', 0) >= 85) or (row.get('ensemble_accuracy', 0) >= 85)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('xgboost_ensemble_performance_summary.csv', index=False)
        print("✅ XGBoost & Ensemble performance summary saved to: xgboost_ensemble_performance_summary.csv")
        
        # Display summary
        print("\n📋 PERFORMANCE SUMMARY:")
        print(f"Total Models Analyzed: {len(summary_df)}")
        print(f"Models with ≥85% accuracy in any category: {summary_df['any_85_plus'].sum()}/{len(summary_df)}")
        print(f"XGBoost ≥85%: {summary_df['xgb_85_plus'].sum()}/{len(summary_df)}")
        print(f"Ensemble ≥85%: {summary_df['ensemble_85_plus'].sum()}/{len(summary_df)}")
    
    print("\n✅ Analysis complete! All results saved.")

def main():
    """Main function to run XGBoost and Ensemble optimization analysis"""
    
    print("🎯 XGBOOST & ENSEMBLE OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print("Focusing on XGBoost and Ensemble models only...")
    print("LSTM is kept in background but not displayed in metrics.")
    
    # Load enhanced models data
    enhanced_models_df = load_xgboost_ensemble_data()
    
    if enhanced_models_df.empty:
        print("❌ No enhanced models found!")
        return
    
    print(f"✅ Loaded data for {len(enhanced_models_df)} crop-mandi combinations")
    
    # Display current status
    print(f"\n📊 ENHANCED MODEL AVAILABILITY:")
    print(f"XGBoost Enhanced: {enhanced_models_df['xgb_enhanced'].sum()}/{len(enhanced_models_df)}")
    print(f"Ensemble Enhanced: {enhanced_models_df['ensemble_enhanced'].sum()}/{len(enhanced_models_df)}")
    
    # Create performance plots (XGBoost & Ensemble only)
    create_xgboost_ensemble_dashboard(enhanced_models_df)
    
    # Create confusion matrix analysis (XGBoost & Ensemble only)
    create_confusion_matrix_xgboost_ensemble(enhanced_models_df)
    
    # Perform detailed analysis (XGBoost & Ensemble only)
    detailed_xgboost_ensemble_analysis(enhanced_models_df)
    
    # Save results
    save_xgboost_ensemble_results(enhanced_models_df)
    
    print("\n🎉 XGBOOST & ENSEMBLE OPTIMIZATION COMPLETE!")
    print("📁 Check the generated files and plots for detailed results.")
    print("💡 LSTM models are still part of the ensemble but not displayed in metrics.")

if __name__ == "__main__":
    main()

