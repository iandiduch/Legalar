"""Allowlist de IPs/CIDR para endpoints admin. Sin dependencias externas: el
modulo estandar `ipaddress` alcanza para una lista chica de rangos confiables
(red de campus, VPN admin) -- no hace falta nada mas sofisticado a esta escala."""

import ipaddress


def is_ip_in_allowlist(ip: str, cidr_csv: str) -> bool:
    if not cidr_csv.strip():
        return True  # allowlist deshabilitado (default) = permite cualquier IP

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False

    for cidr in cidr_csv.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            if addr in ipaddress.ip_network(cidr, strict=False):
                return True
        except ValueError:
            continue
    return False
