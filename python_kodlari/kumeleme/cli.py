import argparse
import json
from pathlib import Path
from threadpoolctl import threadpool_limits

def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description='URF + K-Medoids baseline and mixed-data clustering comparison')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('--config',type=Path,required=True);p.add_argument('--selection',type=Path);p.add_argument('--output',type=Path)
    for name in ['check','report']:
        p=sub.add_parser(name);p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('apply');p.add_argument('--run',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--period')
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command=='run':
            from .study import run
            selection=args.selection or Path(json.loads((root/'outputs/dinamik_feature_selection/latest.json').read_text())['path'])
            settings=json.loads(args.config.read_text())
            if settings.get('feature_selection_ratios'):
                from .parameter_grid import run_grid
                run_grid(selection,args.config,args.output)
            else:run(selection,args.config,args.output)
        elif args.command=='check':
            if (args.run/'grid_plan.json').exists():
                from .parameter_grid import verify_grid
                verify_grid(args.run)
            else:
                from .verify import verify
                verify(args.run)
        elif args.command=='report':
            if (args.run/'grid_plan.json').exists():
                from .parameter_grid import report_grid
                report_grid(args.run)
            else:
                from .report import report
                report(args.run)
        else:
            from .study import apply
            active=args.run
            if (active/'grid_plan.json').exists():
                plan=json.loads((active/'grid_plan.json').read_text())
                active=active/next(c['relative_path'] for c in plan['cases'] if c['case_id']==plan['selected_case_id'])
            apply(active,args.panel,args.output,args.period)

if __name__=='__main__':main()
