import React from 'react';

export function Card({
  title,
  subtitle,
  action,
  children,
  className = '',
  headerClassName = '',
  bodyClassName = '',
  badge,
  icon: Icon = null,
}) {
  return (
    <div className={`tech-card ${className}`}>
      {(title || subtitle || action || badge || Icon) && (
        <div className={`px-5 py-3.5 border-b border-border-subtle/80 flex items-center justify-between gap-4 ${headerClassName}`}>
          <div className="flex items-center gap-2.5">
            {Icon && (
              <div className="w-7 h-7 rounded bg-surface-100 border border-border-subtle flex items-center justify-center text-accent flex-shrink-0">
                <Icon className="w-3.5 h-3.5" />
              </div>
            )}
            <div>
              {title && <h3 className="text-xs font-semibold text-slate-100 tracking-wide font-sans">{title}</h3>}
              {subtitle && <p className="text-[11px] text-slate-400 mt-0.5">{subtitle}</p>}
            </div>
            {badge && <div className="ml-1">{badge}</div>}
          </div>
          {action && <div className="flex items-center gap-2">{action}</div>}
        </div>
      )}
      <div className={`p-5 ${bodyClassName}`}>
        {children}
      </div>
    </div>
  );
}
